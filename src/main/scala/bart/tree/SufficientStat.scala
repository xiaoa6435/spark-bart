package bart.tree

import bart.ParArrOrSeqArr
import bart.discretizer.TreePoint
import bart.sampler.{HalfNormSampler, SigmaSquareSampler}

import scala.collection.mutable

class SufficientStat(
  val availableFeaturesAndBins: Map[Int, Set[Int]],
  val weightedCount: Double,
  val chainId: Int) extends Serializable {

  var rSum: Double = 0.0
  var leafResp: Double = 0.0

  def refreshRSum(targetInput: ParArrOrSeqArr[TreePoint]): Unit = {
    rSum = targetInput.map{point =>
      val leafResp = point.getTmpResp(chainId)
      (point.getResidual(chainId) + leafResp) * point.weight
    }.reduce(_ + _)
  }

  def refreshRSum(left: SufficientStat, right: SufficientStat): Unit = {
    rSum = left.rSum + right.rSum
  }

  def refreshRSum(value: Double): Unit = {
    rSum = value
  }

  def logLikelihood(sigmaSquareSampler: SigmaSquareSampler, tauSampler: HalfNormSampler): Double = {
    val sigmaSquare = sigmaSquareSampler.value
    val tau = tauSampler.value
    0.5 * (math.log(sigmaSquare / (sigmaSquare + tau * weightedCount)) +
      tau * rSum * rSum / sigmaSquare / (sigmaSquare + tau * weightedCount))
  }

  private def leafResp_(sigmaSquareSampler: SigmaSquareSampler, tauSampler: HalfNormSampler) = {
    val likelihoodVar = sigmaSquareSampler.value / weightedCount
    val likelihoodMean = rSum / weightedCount
    val posteriorVar = 1 / (1 / tauSampler.value + 1 / likelihoodVar)
    val posteriorMean = likelihoodMean / likelihoodVar * posteriorVar
    posteriorMean + util.Random.nextGaussian * math.sqrt(posteriorVar)
  }

  def refreshLeafResp(sigmaSquareSampler: SigmaSquareSampler, tauSampler: HalfNormSampler): Unit = {
    leafResp = leafResp_(sigmaSquareSampler, tauSampler)
  }

  def copy(newChainId: Int = chainId): SufficientStat = {
    val newSuff = new SufficientStat(availableFeaturesAndBins, weightedCount, newChainId)
    newSuff.rSum = rSum
    newSuff.leafResp = leafResp
    newSuff
  }

  override def toString: String =
    s"""count: $weightedCount, rSum: ${rSum.formatted("%.2f")}, """ +
      s"""mu: ${(rSum / weightedCount).formatted("%.2f")}, """ +
      s"""leafResp: ${leafResp.formatted("%.2f")}, """
}

case object SufficientStat {
  def calculate(
    targetInput: ParArrOrSeqArr[TreePoint],
    parentAvailableFeaturesAndBins: Map[Int, Set[Int]],
    chainId: Int): SufficientStat = {

    val weightedCountAndRSum = targetInput.
      map { point =>
        val weight = point.weight
        val leafResp = point.getTmpResp(chainId)
        val rSum = (point.getResidual(chainId) + leafResp) * point.weight
        (weight, rSum)
      }.
      reduce((a, b) => (a._1 + b._1, a._2 + b._2))

    val featIds = parentAvailableFeaturesAndBins.keys.toList.sorted.toBuffer
    val featSize = featIds.length
    val _availableFeaturesAndBins = featIds.
      map(featId => featId -> mutable.Set[Int]()).toMap

    targetInput.foreach{point =>
      var i = 0
      while (i < featSize && featIds.contains(i)) {
        _availableFeaturesAndBins(i).add(point.binnedFeatures(i))
        if (_availableFeaturesAndBins(i).size == parentAvailableFeaturesAndBins(i).size) {
          featIds.drop(i)
        }
        i = i + 1
      }
    }

    val availableFeaturesAndBins = _availableFeaturesAndBins.
      map{case (k, v) => k -> v.toSet}.
      filter(kv => kv._2.size > 1)

    val (weightedCount, rSum) = weightedCountAndRSum
    val suff = new SufficientStat(availableFeaturesAndBins, weightedCount, chainId)
    suff.refreshRSum(rSum)
    suff
  }

  def calculate(
    targetInput: ParArrOrSeqArr[TreePoint], parentSuff: SufficientStat): SufficientStat = {
    calculate(targetInput, parentSuff.availableFeaturesAndBins, parentSuff.chainId)
  }
}