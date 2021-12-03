package bart.sampler.tree

import bart.ParArrOrSeqArr
import bart.discretizer.{CategoricalSplit, ContinuousSplit, FindSplits, Split, TreePoint}
import bart.sampler.ForestPara
import bart.tree.{LearningNode, SufficientStat}

case class Proposer(
  originNode: LearningNode,
  proposerNode: LearningNode,
  filteredDf: ParArrOrSeqArr[TreePoint],
  isQualified: Boolean,
  logProbSplitWithinNode: Double,
  logProbAccept: Double,
  numLeaves: Int,
  numPruneNodesAvailable: Int
)

case object Proposer {

  def filterDf(
    df: ParArrOrSeqArr[TreePoint],
    topNode: LearningNode,
    originNode: LearningNode,
    chainId: Int): ParArrOrSeqArr[TreePoint] = {

    val tgtDf = df.filter(p =>
      LearningNode.isBelongTargetNode(topNode, originNode, p.binnedFeatures))

    tgtDf.foreach { point =>
      val oldLeafResp = originNode.predict(point.binnedFeatures)
      point.setTmpResp(oldLeafResp, chainId)
    }
    tgtDf
  }

  def _randomSelectOne[T](arr: Array[T], rng: util.Random): T = {
    val ind = rng.nextInt(arr.length)
    arr(ind)
  }

  def randomPickASplitInNode(node: LearningNode, para: ForestPara, rng: util.Random): (Split, Double) = {

    checkIsSplittable(node)

    val featuresAndSplits = node.sufficientStat.availableFeaturesAndBins
    val availableFeatures = featuresAndSplits.keys
    val probFeatureSelected = -math.log(availableFeatures.size)
    val featId = _randomSelectOne(availableFeatures.toArray, rng)
    val availableBins = featuresAndSplits(featId).toArray.sorted
    val arity = availableBins.length
    val (split, numBins) = if (para.isCategorical(featId)) {
      val numBins = (1 << arity - 1) - 1
      val pickId = rng.nextInt(numBins) + 1
      val _leftInd = FindSplits.extractMultiClassCategories(pickId, numBins)
      val _leftBins = _leftInd.map(id => availableBins(id)).sorted
      val split = new CategoricalSplit(featId, _leftBins.toArray, para.featureArity(featId))
      //val split = new CategoricalSplit(featId, _leftBins.toArray, para.maxArity)
      (split, numBins)
    } else {
      val numBins = arity - 1
      val binId = _randomSelectOne(availableBins.init, rng)
      val split = new ContinuousSplit(featId, -1.0, binId)
      (split, numBins)
    }

    val probSplitSelected = -math.log(numBins)
    (split, probFeatureSelected * probSplitSelected)
  }

  def logProbSplitWithinNode(node: LearningNode): Double = {

    checkIsSplittable(node)

    val featuresAndSplits = node.sufficientStat.availableFeaturesAndBins
    val probFeatureSelected = -math.log(featuresAndSplits.size)
    val split = node.split.get
    val featId = split.featureIndex
    val availableBins = featuresAndSplits(featId)
    val arity = availableBins.size
    val numBins = split match {
      case _: CategoricalSplit =>
        (1 << arity - 1) - 1
      case _: ContinuousSplit =>
        arity - 1
    }
    val probSplitSelected = -math.log(numBins)
    probFeatureSelected + probSplitSelected
  }

  def proposerNodeAndCheck(
    filteredDf: ParArrOrSeqArr[TreePoint],
    node: LearningNode,
    para: ForestPara,
    pickedSplit: Split): (Boolean, LearningNode) = {

    require(node.isLeaf || (node.isSinglyInternal && node.split.nonEmpty),
      "only for singly internal or leaf node")

    val _proposerNode = node.copy()
    val proposerNode = growOrChange(_proposerNode, filteredDf, pickedSplit)
    val minInstances = para.minInstancesPerNode
    val isQualified = proposerNode.split.isDefined &&
      proposerNode.leftChild.isDefined &&
      proposerNode.leftChild.get.sufficientStat.weightedCount >= minInstances &&
      proposerNode.rightChild.isDefined &&
      proposerNode.rightChild.get.sufficientStat.weightedCount >= minInstances
    (isQualified, proposerNode)

  }

  def checkIsSplittable(node: LearningNode): Boolean = {

    require(node.isLeaf || (node.isSinglyInternal && node.split.nonEmpty),
      "only for singly internal or leaf node")

    val availableFeaturesAndBins = node.sufficientStat.availableFeaturesAndBins
    val haveFeat = availableFeaturesAndBins.nonEmpty

//    require(
//      haveFeat,
//      s"no feats: availableFeaturesAndBins: $availableFeaturesAndBins")

    val allFeatCanSplit = !availableFeaturesAndBins.exists(kv => kv._2.size < 2)
//    require(
//      allFeatCanSplit,
//      s"no splits in feat: availableFeaturesAndBins: $availableFeaturesAndBins")

    haveFeat && allFeatCanSplit
  }

  def growOrChange(
    node: LearningNode,
    filteredDf: ParArrOrSeqArr[TreePoint],
    split: Split): LearningNode = {

    checkIsSplittable(node)

    node.split = Some(split)
    node.df = Some(filteredDf)
    val (leftDf, rightDf) = filteredDf.partition{point =>
      split.shouldGoLeft(point.binnedFeatures)
    }

    val leftSuff = SufficientStat.calculate(leftDf, node.sufficientStat)
    node.leftChild = Some(new LearningNode(
      LearningNode.leftChildIndex(node.id),
      None, None, None, leftSuff, Some(leftDf)
    ))

    val rightSuff = SufficientStat.calculate(rightDf, node.sufficientStat)
    node.rightChild = Some(new LearningNode(
      LearningNode.rightChildIndex(node.id),
      None, None, None, rightSuff, Some(rightDf)
    ))
    node.sufficientStat.rSum = leftSuff.rSum + rightSuff.rSum
    node
  }

  def prune(node: LearningNode): LearningNode = {
    assert(node.isSinglyInternal && node.split.nonEmpty,
      "only for singly internal")
    node.leftChild = None
    node.rightChild = None
    node.split = None
    node
  }
}