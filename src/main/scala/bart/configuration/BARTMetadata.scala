package bart.configuration

import bart.sampler.{ForestPara, HalfNormPara, SigmaSquarePara}
import org.apache.commons.math3.distribution.ChiSquaredDistribution

class BARTMetadata(
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
  val numTrees: Int = 200,
  val alpha: Double = 0.95,
  val beta: Double = 2.0,
  val sd: Double = 0.25,
  override val minInstancesPerNode: Int = 1,
  override val probGrow: Double = 2.5 / 9.0,
  override val probPrune: Double = 2.5 / 9.0,
  override val probChange: Double = 4.0 / 9.0,
  override val _lambda: Double = 0.0,
  override val q: Double = 0.9,
  override val nu: Double = 3.0
) extends Metadata {

  private val featIndexes = Range(0, inputSummarizer.numFeatures - 1).toArray

  val forestPara: ForestPara = ForestPara(
    featIndexes, featureArity, chainId = 0, maxBins, maxArity,
    numTrees, alpha, beta, sd, useScale = false, useHalfNorm = false,
    minInstancesPerNode, probGrow, probPrune, probChange
  )

  val lambda: Double = {
    if (_lambda > 0) {
      _lambda
    } else {
      val chiSqDist = new ChiSquaredDistribution(nu)
      val variance = inputSummarizer.labelSummarizer.variance
      val scaledVariance = variance / math.pow(inputSummarizer.minMaxScale, 2)
      chiSqDist.inverseCumulativeProbability(1 - q) / nu * scaledVariance
    }
  }
  val sigmaSquarePara = new SigmaSquarePara(inputSummarizer.numPoints, lambda, q, nu)
  val tauPara = new HalfNormPara(sd, numTrees, false)
}

