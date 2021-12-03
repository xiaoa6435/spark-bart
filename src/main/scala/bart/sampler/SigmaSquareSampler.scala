package bart.sampler

import org.apache.commons.math3.distribution.GammaDistribution

class SigmaSquarePara(
  val numPoints: Long,
  val lambda: Double = 0.0,
  val q: Double = 0.9,
  val nu: Double = 3.0
) extends Para

class SigmaSquareSampler(
  val para: SigmaSquarePara,
  val chainId: Int
) extends Sampler {
  type T = Double
  type A = Double

  var value: T = 1.0

  def update(residualsSquareSum: A): Unit = {
    val nu = (para.nu + para.numPoints) / 2.0
    val lambda = (para.nu * para.lambda + residualsSquareSum) / 2.0
    value = 1.0 / new GammaDistribution(nu, 1.0 / lambda).sample()
  }

  def copy(newChainId: Int): SigmaSquareSampler = {

    val newSampler = new SigmaSquareSampler(para, newChainId)
    newSampler.value = value
    newSampler
  }

  override def toString: String = {
    s"chainId: $chainId, sigmaSquare: ${value.formatted("%.5f")}, "
  }
}