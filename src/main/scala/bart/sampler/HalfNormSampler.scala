package bart.sampler

import org.apache.commons.math3.distribution.GammaDistribution

class HalfNormPara(
  val sd: Double,
  val numTrees: Int,
  val useHalfNorm: Boolean) extends Para

class HalfNormSampler(
  val para: HalfNormPara,
  val chainId: Int
) extends Sampler {

  type T = Double
  type A = ForestSampler
  val init: T = {
    para.sd / math.sqrt(para.numTrees) / (if (para.useHalfNorm) 0.674 else 1.0)
  }
  var value: T = math.pow(init, 2)
  def update(forestSampler: A): Unit = {
    val leafRespSqrSumAndSize = forestSampler.leafRespSqrSumAndSize
    if (para.useHalfNorm) {
      val (leafRespSqrSum, leafNodeSize) = leafRespSqrSumAndSize
      val shape = (1.0 + leafRespSqrSum / math.pow(value, 2)) / 2
      val delta = new GammaDistribution((1.0 + leafNodeSize) / 2, shape).sample()
      value = math.pow(delta * init, 2)
    }
  }

  def copy(newChainId: Int): HalfNormSampler = {
    val newSampler = new HalfNormSampler(para, newChainId)
    newSampler.value = value
    newSampler
  }

  override def toString: String = {
    s"chainId: $chainId, tau: ${value.formatted("%.5f")}, "
  }
}
