package bart.sampler

import bart.{ParArr, ParArrOrSeqArr, SeqArr}
import bart.discretizer.TreePoint
import org.apache.commons.math3.distribution.NormalDistribution

class ScalesPara (
  val bScalePrec: Double = 2,
  val muScalePrec: Double = 1.0,
  val useBScale: Boolean = true,
  val useMuScale: Boolean = true
) extends Para {
  val precs: Array[Double] = Array(muScalePrec, bScalePrec, bScalePrec)
}

class ScalesSampler(
  val para: ScalesPara,
  val df: ParArrOrSeqArr[TreePoint],
  val chainId: Int
) extends Sampler {

  type T = Array[Double]
  type A = SigmaSquareSampler

  var value: T = Array(1.0, -0.5, 0.5)
  var oldValue: T = value
  val numScales: Int = value.length

  def update(sigmaSquareSampler: A): Unit = {
    //val (df, sigmaSquare) = dfAndSigmaSquare
    oldValue = value
    val sigmaSquare = sigmaSquareSampler.value
    val Array(oldMuScale, oldB0Scale, oldB1Scale) = value

    val stats = df.aggregate(Array.fill(numScales * 2)(0.0))(
      seqop = (acc, p) => {
        val residual = p.getResidual(chainId)
        if (para.useMuScale) {
          val conResp = p.getConResp(chainId)
          val mw = p.weight * math.pow(conResp / oldMuScale, 2) / sigmaSquare
          val mrw = p.weight * conResp * (residual + conResp) / sigmaSquare / oldMuScale
          acc(0) = mw
          acc(3) = mrw
        }

        if (para.useBScale) {
          val modResp = p.getModResp(chainId)
          if (p.binnedFeatures.last == 0) {
            val bw = p.weight * math.pow(modResp / oldB0Scale, 2) / sigmaSquare
            val brw = p.weight * modResp * (residual + modResp) / sigmaSquare / oldB0Scale
            acc(1) = bw
            acc(4) = brw
          } else {
            val bw = p.weight * math.pow(modResp / oldB1Scale, 2) / sigmaSquare
            val brw = p.weight * modResp * (residual + modResp) / sigmaSquare / oldB1Scale
            acc(2) = bw
            acc(5) = brw
          }
        }
        acc
      },
      combop = (acc1, acc2) => acc1.zip(acc2).map(x => x._1 + x._2)
    )
    val wws = stats.slice(0, numScales)
    val rws = stats.slice(numScales, 2 * numScales)

    val newValue = wws.zip(rws).zip(para.precs).
      map{case ((ww, rw), prec) =>
        val fcVar = 1.0 / (ww + prec)
        new NormalDistribution(fcVar * rw, math.sqrt(fcVar)).sample()
      }

    if (para.useMuScale) {
      value(0) = newValue(0)
    }

    if (para.useBScale) {
      value(1) = newValue(1)
      value(2) = newValue(2)
    }
  }

  def copy(newChainId: Int): ScalesSampler = {
    val _df = df match {
      case df: ParArr[TreePoint] =>
        new SeqArr(df.df.toArray)
      case _ =>
        df
    }
    val newSampler = new ScalesSampler(para, _df, newChainId)
    newSampler.value = value
    newSampler
  }

  override def toString: String = {
    s"chainId: $chainId, scales: ${value.toList.map(_.formatted("%.5f"))}, "
  }
}
