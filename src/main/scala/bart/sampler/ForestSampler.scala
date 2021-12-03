package bart.sampler

import bart.discretizer.TreePoint
import bart.sampler.tree.{ChangeProposer, GrowProposer, PruneProposer, TreeMutationProposer}
import bart.tree.{LearningNode, SufficientStat}
import bart.{ParArr, ParArrOrSeqArr, SeqArr}

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