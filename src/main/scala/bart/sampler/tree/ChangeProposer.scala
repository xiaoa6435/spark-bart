package bart.sampler.tree

import bart.ParArrOrSeqArr
import bart.discretizer.TreePoint
import bart.sampler.{ForestPara, HalfNormSampler, SigmaSquareSampler}
import bart.tree.LearningNode

class ChangeProposer(
  val topNode: LearningNode,
  val para: ForestPara,
  val proposer: Proposer,
  val treeId: Int,
  val sigmaSquareSampler: SigmaSquareSampler,
  val tauSampler: HalfNormSampler
) extends TreeMutationProposer {
  val kind: String = "change"
  def logTransitionRatio: Double = 0.0

  override def logLikelihoodRatio: Double = {
    val newNode = proposer.proposerNode
    val newLeft = newNode.leftChild.get.sufficientStat.logLikelihood(sigmaSquareSampler, tauSampler)
    val newRight = newNode.rightChild.get.sufficientStat.logLikelihood(sigmaSquareSampler, tauSampler)

    val oldNode = proposer.originNode
    val oldLeft = oldNode.leftChild.get.sufficientStat.logLikelihood(sigmaSquareSampler, tauSampler)
    val oldRight = oldNode.rightChild.get.sufficientStat.logLikelihood(sigmaSquareSampler, tauSampler)
    newLeft + newRight - oldLeft - oldRight
  }

  override def logTreeRatio: Double = 0.0

  override def updateSuffStat(): Unit = {
    //refreshRSum(proposer.proposerNode)
    refreshRSum(proposer.originNode)
  }
}

object ChangeProposer {
  def apply(
    df: ParArrOrSeqArr[TreePoint], topNode: LearningNode,
    para: ForestPara, treeId: Int,
    sigmaSquareSampler: SigmaSquareSampler,
    tauSampler: HalfNormSampler
  ): ChangeProposer = {
    val proposer = generate(df, topNode, para)
    new ChangeProposer(topNode, para, proposer, treeId, sigmaSquareSampler, tauSampler)
  }

  def generate(df: ParArrOrSeqArr[TreePoint], topNode: LearningNode, para: ForestPara): Proposer = {

    val allSinglyInternalNodes = topNode.allSinglyInternalNodes
    if (allSinglyInternalNodes.isEmpty) {
      df.foreach { point =>
        val oldLeafResp = topNode.predict(point.binnedFeatures)
        point.setTmpResp(oldLeafResp, topNode.sufficientStat.chainId)
      }
      return Proposer(topNode.deepCopy(para.chainId), topNode.deepCopy(para.chainId), df, isQualified = false, 0.0, 0.0, 0, 0)
    }

    val rng = new util.Random()
    val _originNode = Proposer._randomSelectOne(allSinglyInternalNodes, rng)
    val originNode = _originNode.deepCopy(para.chainId)
    val filteredDf = Proposer.filterDf(df, topNode, originNode, para.chainId)
    if (!Proposer.checkIsSplittable(_originNode)) {
      return Proposer(
        originNode, _originNode.copy(), filteredDf,
        isQualified = false, -1.0, -1.0, -1, -1
      )
    }

    val (pickedSplit, _) = Proposer.randomPickASplitInNode(originNode, para, rng)
    val (isQualified, proposerNode) = Proposer.proposerNodeAndCheck(
      filteredDf, originNode, para, pickedSplit)

    val logProbAccept = math.log(rng.nextDouble)
    Proposer(
      originNode, proposerNode, filteredDf,
      isQualified, -1.0, logProbAccept, -1, -1)
  }
}
