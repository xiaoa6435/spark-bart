package bart.sampler.tree

import bart.sampler.{ForestPara, HalfNormSampler, SigmaSquareSampler}
import bart.tree.LearningNode

trait TreeMutationProposer extends Serializable {

  val topNode: LearningNode
  val para: ForestPara
  val proposer: Proposer
  val treeId: Int
  val sigmaSquareSampler: SigmaSquareSampler
  val tauSampler: HalfNormSampler
  val chainId: Int = para.chainId
  val kind: String

  assert(
    sigmaSquareSampler.chainId == chainId && sigmaSquareSampler.chainId == tauSampler.chainId,
    "sigma, tau and forest sampler should have then same chainId"
  )

  def logTransitionRatio: Double
  def logLikelihoodRatio: Double
  def logTreeRatio: Double

  def logProbRatio: Double = logTransitionRatio + logLikelihoodRatio + logTreeRatio
  def isAccept: Boolean = proposer.isQualified && (logProbRatio > proposer.logProbAccept)
  def acceptedNode: LearningNode = if (isAccept) proposer.proposerNode else proposer.originNode

  def update(): Unit = {
    updateSuffStat()
    refreshLeafRespAndUpdateResidual(acceptedNode)
    updateTree(acceptedNode)
    thin(proposer.proposerNode)
    thin(proposer.originNode)
  }

  def updateSuffStat(): Unit

  def refreshLeafRespAndUpdateResidual(acceptedNode: LearningNode): Unit = {

    val filteredDf = proposer.filteredDf
    require(acceptedNode.isSinglyInternal || acceptedNode.isLeaf,
      "only for singly internal or leaf node")

    if (acceptedNode.isLeaf) {
      acceptedNode.sufficientStat.refreshLeafResp(sigmaSquareSampler, tauSampler)
      val newLeafResp = acceptedNode.sufficientStat.leafResp
      filteredDf.foreach { point =>
        val leafResp = point.getTmpResp(chainId)
        point.setResidual(point.getResidual(chainId) + leafResp - newLeafResp, chainId)
      }
    } else {
      val left = acceptedNode.leftChild.get.sufficientStat
      val right = acceptedNode.rightChild.get.sufficientStat
      left.refreshLeafResp(sigmaSquareSampler, tauSampler)
      right.refreshLeafResp(sigmaSquareSampler, tauSampler)
      val split = acceptedNode.split.get
      filteredDf.foreach { point =>
        val binnedFeatures = point.binnedFeatures
        val newLeafResp = if (split.shouldGoLeft(binnedFeatures)) {
          left.leafResp
        } else {
          right.leafResp
        }
        val leafResp = point.getTmpResp(chainId)
        point.setResidual(point.getResidual(chainId) + leafResp - newLeafResp, chainId)
      }
    }
  }

  def updateTree(acceptedNode: LearningNode): LearningNode = {
    val parentNodeId = LearningNode.parentIndex(acceptedNode.id)
    if (parentNodeId == 0) {
      topNode.split = acceptedNode.split
      topNode.leftChild = acceptedNode.leftChild
      topNode.rightChild = acceptedNode.rightChild
      topNode.sufficientStat.rSum = acceptedNode.sufficientStat.rSum
      topNode.sufficientStat.leafResp = acceptedNode.sufficientStat.leafResp
      return topNode
    }

    val parent = LearningNode.getNode(parentNodeId, topNode)
    if (LearningNode.isLeftChild(acceptedNode.id)) {
      parent.leftChild = Some(acceptedNode)
    } else {
      parent.rightChild = Some(acceptedNode)
    }
    topNode
  }

  def logTransitionRatioForGrowOrPrune(): Double = {
    val numLeaves = proposer.numLeaves
    val numPruneNodes = proposer.numPruneNodesAvailable
    val logProbSplitWithinNode = proposer.logProbSplitWithinNode
    val toPrune = math.log(para.probPrune / numPruneNodes)
    val toGrow = math.log(para.probGrow / numLeaves) + logProbSplitWithinNode
    toPrune - toGrow
  }

  def logTreeRatioForGrowOrPrune(): Double = {
    val depth = LearningNode.indexToLevel(proposer.originNode.id)
    val logProbSplitWithinNode = proposer.logProbSplitWithinNode
    val probLeftNotSplit = logProbNodeNotSplit(depth + 1)
    val probRightNotSplit = logProbNodeNotSplit(depth + 1)
    val probParentSplit = logProbNodeSplit(depth)
    val probParentNotSplit = logProbNodeNotSplit(depth)
    val probSplitChosen = logProbSplitWithinNode

    probLeftNotSplit + probRightNotSplit + probParentSplit -
      probParentNotSplit + probSplitChosen
  }

  def thin(node: LearningNode): Unit = {
    require(
      node.isSinglyInternal || node.isLeaf, "only for Singly internal or leaf node")
    node.df = None
    if (node.isSinglyInternal) {
      node.leftChild.get.df = None
      node.rightChild.get.df = None
    }
  }

  def logProbNodeSplit(depth: Int): Double = {
    math.log(para.alpha * math.pow(1 + depth, -para.beta))
  }

  def logProbNodeNotSplit(depth: Int): Double = {
    math.log(1.0 - para.alpha * math.pow(1 + depth, -para.beta))
  }

  def refreshRSum(node: LearningNode): Unit = {
    require(
      node.isSinglyInternal || node.isLeaf,
      "only for Singly internal or leaf node")

    val filteredDf = proposer.filteredDf
    if (node.isLeaf) {
      node.sufficientStat.refreshRSum(filteredDf)
      return
    }

    val left = node.leftChild.get
    val right = node.rightChild.get
    if (node.df.isEmpty || left.df.isEmpty || right.df.isEmpty) {
      node.df = Some(filteredDf)
      val split = node.split.get
      val (leftDf, rightDf) = filteredDf.partition{point => split.shouldGoLeft(point.binnedFeatures)}
      left.df = Some(leftDf)
      right.df = Some(rightDf)
    }

    val leftSuff = left.sufficientStat
    leftSuff.refreshRSum(left.df.get)

    val rightSuff = right.sufficientStat
    rightSuff.refreshRSum(right.df.get)
    node.sufficientStat.refreshRSum(leftSuff, rightSuff)
  }

  override def toString: String = {
    if (!proposer.isQualified) {
      s"kind: $kind, not valid"
    } else {
      s"""kind: $kind, isAccept: $isAccept, R: ${logProbRatio.formatted("%.3f")}, """ +
        s"""TR: ${logTransitionRatio.formatted("%.3f")}， LL: ${logLikelihoodRatio.formatted("%.3f")}, """ +
        s"""TR: ${logTreeRatio.formatted("%.3f")}"""
    }
  }
}
