package bart.sampler.tree

import bart.ParArrOrSeqArr
import bart.discretizer.TreePoint
import bart.sampler.{ForestPara, HalfNormSampler, SigmaSquareSampler}
import bart.tree.LearningNode

class PruneProposer(
  val topNode: LearningNode,
  val para: ForestPara,
  val proposer: Proposer,
  val treeId: Int,
  val sigmaSquareSampler: SigmaSquareSampler,
  val tauSampler: HalfNormSampler
) extends TreeMutationProposer {

  val kind: String = "prune"

  def logTransitionRatio: Double = -logTransitionRatioForGrowOrPrune

  override def logLikelihoodRatio: Double = {
    val node = proposer.originNode
    val left = node.leftChild.get.sufficientStat.logLikelihood(sigmaSquareSampler, tauSampler)
    val right = node.rightChild.get.sufficientStat.logLikelihood(sigmaSquareSampler, tauSampler)
    val parent = node.sufficientStat.logLikelihood(sigmaSquareSampler, tauSampler)
    parent - left - right
  }

  override def logTreeRatio: Double = -logTreeRatioForGrowOrPrune

  override def updateSuffStat(): Unit = {
    refreshRSum(proposer.originNode)
    proposer.proposerNode.sufficientStat.refreshRSum(proposer.originNode.sufficientStat.rSum)
  }
}

object PruneProposer {
  def apply(
    df: ParArrOrSeqArr[TreePoint],
    topNode: LearningNode,
    para: ForestPara,
    treeId: Int,
    sigmaSquareSampler: SigmaSquareSampler,
    tauSampler: HalfNormSampler
  ): PruneProposer = {
    val proposer = generate(df, topNode, para)
    new PruneProposer(topNode, para, proposer, treeId, sigmaSquareSampler, tauSampler)
  }

  def generate(df: ParArrOrSeqArr[TreePoint], topNode: LearningNode, para: ForestPara): Proposer = {
    val allSinglyInternalNodes = topNode.allSinglyInternalNodes
    if (allSinglyInternalNodes.isEmpty) {
      df.foreach { point =>
        val oldLeafResp = topNode.predict(point.binnedFeatures)
        point.setTmpResp(oldLeafResp, topNode.sufficientStat.chainId)
      }
      return Proposer(topNode.copy(), topNode.copy(), df, isQualified = false, -1.0, -1.0, -1, -1)
    }
    val rng = new util.Random()
    val _originNode = Proposer._randomSelectOne(allSinglyInternalNodes, rng)
    //Proposer.checkIsSplittable(_originNode)
    val originNode = _originNode.deepCopy(para.chainId)

    val _proposerNode = originNode.deepCopy(para.chainId)
    val proposerNode = Proposer.prune(_proposerNode)

    val filteredDf = Proposer.filterDf(df, topNode, originNode, para.chainId)
    val logProbSplitWithinNode = Proposer.logProbSplitWithinNode(originNode)
    val logProbAccept = math.log(rng.nextDouble)

    val (numLeaves, numPruneNodes) = {
      (topNode.allLeafNodes.length - 1, topNode.allSinglyInternalNodes.length)
    }

    Proposer(
      originNode, proposerNode, filteredDf,
      isQualified = true, logProbSplitWithinNode, logProbAccept,
      numLeaves, numPruneNodes
    )
  }
}
