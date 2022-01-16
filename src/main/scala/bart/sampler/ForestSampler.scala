package bart.sampler

import bart.discretizer.TreePoint
import bart.sampler.tree.{ChangeProposer, GrowProposer, PruneProposer, TreeMutationProposer}
import bart.tree.{LearningNode, SufficientStat}
import bart.{ParArr, ParArrOrSeqArr, SeqArr}
import breeze.stats.distributions.{Exponential, Uniform}
import org.apache.commons.math3.distribution.NormalDistribution

case class ForestPara (
  featIndexes: Array[Int],
  featureArity: Map[Int, Int],
  chainId: Int,
  maxBins: Int,
  maxArity: Int,
  numTrees: Int,
  alpha: Double,
  beta: Double,
  sd: Double,
  useScale: Boolean,
  useHalfNorm: Boolean,
  minInstancesPerNode: Int,
  probGrow: Double,
  probPrune: Double,
  probChange: Double
) extends Para {
  def isCategorical(featureIndex: Int): Boolean = featureArity.contains(featureIndex)
}

class ForestSampler(
  val para: ForestPara,
  val df: ParArrOrSeqArr[TreePoint]
) extends Sampler {
  override val chainId: Int = para.chainId
  type T = Array[LearningNode]
  type A = (SigmaSquareSampler, HalfNormSampler)

  var value: T = Range(0, para.numTrees).toArray.
      map{_ =>
        val parentAvailableSplits = para.featIndexes.zipWithIndex.map{
          case (featId, i) =>
            val numBin = if (para.isCategorical(i)) para.maxArity else para.maxBins
            featId -> Range(0, numBin).toSet
        }.toMap
        val sufficientStat: SufficientStat = SufficientStat.calculate(
          df, parentAvailableSplits, chainId)
        LearningNode(1, sufficientStat)
      }

  def update(data: A): Unit = {
    val (sigmaSquareSampler, tauSampler) = data
    value.zipWithIndex.foreach({topNodeAndTreeId =>
      val (topNode, treeId) = topNodeAndTreeId
      val proposer = proposerSampler(topNode, treeId, sigmaSquareSampler, tauSampler)
      proposer.update()
    })
  }

  def predict(point: TreePoint): Double = {
    value.map(tree => tree.predict(point.binnedFeatures)).sum
  }

//  def sampleZi(z: Double, y: Double): Double = {
//    val u = util.Random.nextDouble
//    val g = new NormalDistribution
//    if (y.round == 1L) {
//      z + g.inverseCumulativeProbability((1 - u) * g.cumulativeProbability(-z) + u)
//    } else {
//      z - g.inverseCumulativeProbability((1 - u) * g.cumulativeProbability(z) + u)
//    }
//  }

  def sampleZi(z: Double, y: Double): Double = {
    if (y.round == 1L) {
      rtnorm(z, lower = 0.0)
    } else {
      rtnorm(z, upper = 0.0)
    }
  }

  def updateResidual(): Unit = {
    df.foreach{point =>
      val oldResp = predict(point)
      val newResp = sampleZi(oldResp, point.label)
      //point.setResidual(point.getResidual(chainId) + oldResp - newResp, chainId)
      point.setResidual(point.getResidual(chainId) + newResp - oldResp, chainId)
    }
  }

  def rtnorm(
    m: Double = 0,
    sd: Double = 1,
    lower: Double = Double.NegativeInfinity,
    upper: Double = Double.PositiveInfinity): Double = {
    require(upper > lower, "upper must gt lower")
    require(sd != 0, "sd should not equal 0")

    val (l_s, u_s, is_mirror) = if (upper <= m) {
      (-(upper - m) / sd, -(lower - m) / sd, true)
    } else {
      ((lower - m) / sd, (upper - m) / sd, false)
    }
    var r = 0.0
    if (l_s < 0.0 && u_s > 0.0 && u_s - l_s > math.sqrt(2 * math.Pi)) {
      val rngNorm = new NormalDistribution(0, 1)
      r = rngNorm.sample()
      while (!(r >= l_s && r < u_s)) {
        r = rngNorm.sample()
      }
      return r + sd * m
    }

    val ls2p4 = math.sqrt(l_s * l_s + 4)
    val a_double = l_s + ls2p4
    val a = a_double / 2.0
    val l1subls2p4 = math.sqrt(math.exp(1)) / a_double
    if (l_s >= 0.0 && (u_s > l_s + 2 * l1subls2p4 * math.exp((l_s * 2 - l_s * ls2p4) / 4))) {
      val rngUnif = new Uniform(0, 1)
      val rngExp = new Exponential(a)
      r = rngExp.sample() + l_s
      while (!(rngUnif.sample() <= math.exp(-(r - a) * (r - a) / 2) && r < u_s)) {
        r = rngExp.sample() + l_s
      }
    } else {
      val rngUnif_lu = new Uniform(l_s, u_s)
      val rngUnif_0l = new Uniform(0.0, 1.0)
      r = rngUnif_lu.sample()
      var rho = if (l_s > 0) math.exp((l_s * l_s - r * r) / 2) else math.exp(-r * r / 2)
      while (rngUnif_0l.sample() > rho) {
        r = rngUnif_lu.sample()
        rho = if (l_s > 0) math.exp((l_s * l_s - r * r) / 2) else math.exp(-r * r / 2)
      }
    }
    (if (is_mirror) -r else r) * sd + m
  }

  // for debug
  var currTreeMutationProposer: TreeMutationProposer = _
  def proposerSampler(
    topNode: LearningNode, treeId: Int,
    sigmaSquareSampler: SigmaSquareSampler,
    tauSampler: HalfNormSampler
  ): TreeMutationProposer = {
    val rng = util.Random.nextDouble
    val proposer = rng match {
      case _ if rng < para.probGrow =>
        GrowProposer(df, topNode, para, treeId, sigmaSquareSampler, tauSampler)
      case _ if rng < para.probGrow + para.probPrune =>
        PruneProposer(df, topNode, para, treeId, sigmaSquareSampler, tauSampler)
      case _ =>
        ChangeProposer(df, topNode, para, treeId, sigmaSquareSampler, tauSampler)
    }
    currTreeMutationProposer = proposer
    proposer
  }

  def leafRespSqrSumAndSize: (Double, Int) = {
    value.map{iTree =>
      val allLeafNodesSqr = iTree.allLeafNodes.map{r =>
        math.pow(r.sufficientStat.leafResp , 2)}

      val leafNodeSize = allLeafNodesSqr.length
      (allLeafNodesSqr.sum, leafNodeSize)

    }.reduce((a, b) => (a._1 + b._1, a._2 + b._2))
  }

  def residualsSquareSum: Double = df.
    map(r => math.pow(r.getResidual(chainId), 2)).reduce(_ + _)

  def copy(newChainId: Int): ForestSampler = {

    df.foreach(p => p.setResidual(p.getResidual(chainId), newChainId))
    df.foreach(p => p.setTmpResp(p.getTmpResp(chainId), newChainId))

    if (df.head.resp.length / df.head.chainCnt == 5) {
      df.foreach(p => p.setTmpWeight(p.getTmpWeight(chainId), newChainId))
      df.foreach(p => p.setConResp(p.getConResp(chainId), newChainId))
      df.foreach(p => p.setModResp(p.getModResp(chainId), newChainId))
    }

    val _df = df match {
      case df: ParArr[TreePoint] =>
        new SeqArr(df.df.toArray)
      case _ =>
        df
    }

    val newForestSampler = new ForestSampler(para.copy(chainId = newChainId), _df)
    newForestSampler.value = value.map{topNode => topNode.deepCopy(newChainId)}
    newForestSampler
  }

  override def toString: String = {
    s"""chainId: $chainId, residualsSquareSum: ${residualsSquareSum.formatted("%.5f")}"""
  }
}