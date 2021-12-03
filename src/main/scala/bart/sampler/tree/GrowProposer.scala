package bart.sampler.tree

import bart.ParArrOrSeqArr
import bart.discretizer.TreePoint
import bart.sampler.{ForestPara, HalfNormSampler, SigmaSquareSampler}
import bart.tree.LearningNode

class GrowProposer(
  val topNode: LearningNode,
  val para: ForestPara,
  val proposer: Proposer,
  val treeId: Int,
  val sigmaSquareSampler: SigmaSquareSampler,
  val tauSampler: HalfNormSampler
) extends TreeMutationProposer {

  val kind: String = "grow"
  def logTransitionRatio: Double = logTransitionRatioForGrowOrPrune()

  override def logLikelihoodRatio: Double = {
    val node = proposer.proposerNode
    val left = node.leftChild.get.sufficientStat.logLikelihood(sigmaSquareSampler, tauSampler)
    val right = node.rightChild.get.sufficientStat.logLikelihood(sigmaSquareSampler, tauSampler)
    val parent = node.sufficientStat.logLikelihood(sigmaSquareSampler, tauSampler)
    left + right - parent
  }

  override def logTreeRatio: Double = logTreeRatioForGrowOrPrune()

  override def updateSuffStat(): Unit = {
    //refreshRSum(proposer.proposerNode)
    proposer.originNode.sufficientStat.refreshRSum(proposer.proposerNode.sufficientStat.rSum)
  }
}

object GrowProposer {
  def apply(
    df: ParArrOrSeqArr[TreePoint], topNode: LearningNode,
    para: ForestPara, treeId: Int,
    sigmaSquareSampler: SigmaSquareSampler,
    tauSampler: HalfNormSampler
  ): GrowProposer = {
    val proposer = generate(df, topNode, para)
    new GrowProposer(topNode, para, proposer, treeId, sigmaSquareSampler, tauSampler)
  }

  def generate(
    df: ParArrOrSeqArr[TreePoint], topNode: LearningNode,
    para: ForestPara): Proposer = {

    val allLeafNodes = topNode.allLeafNodes

    val rng = new util.Random()
    val _originNode = Proposer._randomSelectOne(allLeafNodes, rng)

    val originNode = _originNode.deepCopy(para.chainId)
    val filteredDf = Proposer.filterDf(df, topNode, originNode, para.chainId)
    //Proposer.checkIsSplittable(_originNode)
    if (!Proposer.checkIsSplittable(_originNode)) {
      return Proposer(
        originNode, _originNode.copy(), filteredDf,
        isQualified = false, -1.0, -1.0, -1, -1
      )
    }

    val (pickedSplit, logProbSplitWithinNode) =
      Proposer.randomPickASplitInNode(originNode, para, rng)
    val (isQualified, proposerNode) = Proposer.proposerNodeAndCheck(
      filteredDf, originNode, para, pickedSplit)

    val logProbAccept = math.log(rng.nextDouble)
    val parentNodeId = LearningNode.parentIndex(originNode.id)

    val (numLeaves, numPruneNodes) =
      if (parentNodeId == 0) {
        (allLeafNodes.length, 1)
      } else {
        val parentNode = LearningNode.getNode(parentNodeId, topNode)

        val numPruneNodes = if (parentNode.isSinglyInternal) {
          topNode.allSinglyInternalNodes.length
        } else {
          topNode.allSinglyInternalNodes.length + 1
        }
        (allLeafNodes.length, numPruneNodes)
      }

    Proposer(
      originNode, proposerNode, filteredDf,
      isQualified, logProbSplitWithinNode, logProbAccept,
      numLeaves, numPruneNodes
    )
  }
}
