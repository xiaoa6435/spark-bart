package bart.configuration

import bart.sampler.{ForestPara, HalfNormPara, ScalesPara, SigmaSquarePara}
import org.apache.commons.math3.distribution.ChiSquaredDistribution

class BCFMetadata(
  val inputSummarizer: InputSummarizer,
  override val categoryFeatureIndexes: Array[Int] = Array.emptyIntArray,
  override val categoryFeatureArity: Array[Int] = Array.emptyIntArray,

  override val maxBins: Int = 32,
  override val maxArity: Int = 5,
  override val chainCnt: Int = 1,
  override val parallelChainCnt: Int = 1,
  override val numBurn: Int = 1000,
  override val numSim: Int = 1000,
  override val numThin: Int = 1,

  private val _controlFeatIndexes: Array[Int] = Array.emptyIntArray,
  private val _moderateFeatIndexes: Array[Int] = Array.emptyIntArray,
  private val _treatFeatIndex: Int = -1,

  val conNumTrees: Int = 200,
  val conAlpha: Double = 0.95,
  val conBeta: Double = 2.0,
  val conSd: Double = 2.0,
  val conUseScale: Boolean = true,
  val conUseHalfNorm: Boolean = true,

  val modNumTrees: Int = 50,
  val modAlpha: Double = 0.25,
  val modBeta: Double = 3.0,
  val modSd: Double = 1.0,
  val modUseScale: Boolean = true,
  val modUseHalfNorm: Boolean = false,

  override val minInstancesPerNode: Int = 1,
  override val probGrow: Double = 2.5 / 9.0,
  override val probPrune: Double = 2.5 / 9.0,
  override val probChange: Double = 4.0 / 9.0,

  val bScalePrec: Double = 2,
  val muScalePrec: Double = 1.0,
  val useBScale: Boolean = true,
  val useMuScale: Boolean = true,

  override val _lambda: Double = 0.0,
  override val q: Double = 0.9,
  override val nu: Double = 3.0
) extends Metadata {

  /** by default, the last col is treat */
  val treatFeatIndex: Int = if (_treatFeatIndex >= 0 && _treatFeatIndex < inputSummarizer.numFeatures) {
    _treatFeatIndex
  } else {
    inputSummarizer.numFeatures - 1
  }

  val controlFeatIndexes: Array[Int] = if (_controlFeatIndexes.nonEmpty) {
    _controlFeatIndexes
  } else {
    Range(0, inputSummarizer.numFeatures - 1).toArray.
      filter(i => i != treatFeatIndex).
      map(i => if (i > treatFeatIndex) i - 1 else i)
  }

  val moderateFeatIndexes: Array[Int] = if (_moderateFeatIndexes.nonEmpty) {
    _moderateFeatIndexes
  } else {
    Range(0, inputSummarizer.numFeatures - 1).toArray.
      filter(i => i != treatFeatIndex).
      map(i => if (i > treatFeatIndex) i - 1 else i)
  }
  //val conCategoryFeatureIndexes: Set[Int] = controlFeatIndexes.filter(isCategorical).toSet
  val conForestPara: ForestPara = ForestPara(
    controlFeatIndexes, featureArity, 0, maxBins, maxArity,
    conNumTrees, conAlpha, conBeta, conSd, conUseScale, conUseHalfNorm,
    minInstancesPerNode, probGrow, probPrune, probChange
  )

  //val modCategoryFeatureIndexes: Set[Int] = moderateFeatIndexes.filter(isCategorical).toSet
  val modForestPara: ForestPara = ForestPara(
    moderateFeatIndexes, featureArity, 0, maxBins, maxArity,
    modNumTrees, modAlpha, modBeta, modSd, modUseScale, modUseHalfNorm,
    minInstancesPerNode, probGrow, probPrune, probChange
  )

  val scalePara = new ScalesPara(bScalePrec, muScalePrec, useBScale, useMuScale)
  val lambda: Double = if (_lambda > 0.0) {
    _lambda
  } else {
    val chiSqDist = new ChiSquaredDistribution(nu)
    chiSqDist.inverseCumulativeProbability(1 - q) / nu
  }
  val sigmaSquarePara = new SigmaSquarePara(
    inputSummarizer.labelSummarizer.weightedCount.toInt,
    //math.sqrt(inputSummarizer.labelSummarizer.variance),
    lambda, q, nu)

  val conTauPara = new HalfNormPara(conSd, conNumTrees, conUseHalfNorm)
  val modTauPara = new HalfNormPara(modSd, modNumTrees, modUseHalfNorm)

}
