package bart.sampler

import bart.ParArrOrSeqArr
import bart.discretizer.TreePoint

class Chain (
  val forest: ForestSampler,
  val sigmaSquare: SigmaSquareSampler,
  val halfNorm: HalfNormSampler,
  val scale: Double,
  val parallelChainId: Int
) extends Serializable {

  sigmaSquare.update(forest.residualsSquareSum)

  def sample(i: Int = 0): Unit = {
    forest.update((sigmaSquare, halfNorm))
    val rmse = math.pow(forest.residualsSquareSum * scale, 0.5)
    println(
      s"parallelChainId: $parallelChainId, chainId: ${forest.chainId}, i: $i " +
        forest.df.head.resp.map(r => r.formatted("%+4.3f")).toList.toString +
        s" rmse: ${rmse.formatted("%+4.3f")}")
    sigmaSquare.update(forest.residualsSquareSum)
  }

  def run(numBurn: Int): Unit = {
    var i = 0
    while (i < numBurn) {
      sample(i)
      i = i + 1
    }
  }

  def run(numSim: Int, numThin: Int, splits: Array[Array[bart.discretizer.Split]]): Array[Array[bart.tree.Node]] = {
    val allSimCnt = numSim * numThin
    val numTrees = forest.value.length
    val resForest = Array.ofDim[bart.tree.Node](numSim, numTrees)
    var i = 0
    while (i < allSimCnt) {
      sample(i)
      if (i % numThin == 0) {
        val res = forest.value.map(_.toNode(splits))
        resForest(i / numThin) = res
      }
      i = i + 1
    }
    resForest
  }

  def copy(newChainId: Int): Chain = {
    val newForest = forest.copy(newChainId)
    val newSigmaSquare = sigmaSquare.copy(newChainId)
    val newHalfNorm = halfNorm.copy(newChainId)
    new Chain(newForest, newSigmaSquare, newHalfNorm, scale, parallelChainId)
  }
}

object Chain {
  def apply(
    df: ParArrOrSeqArr[TreePoint],
    metadata: bart.configuration.BARTMetadata,
    parallelChainId: Int
  ): Chain = {
    val chainId = metadata.forestPara.chainId
    val forest = new ForestSampler(metadata.forestPara, df)
    val sigmaSquare = new SigmaSquareSampler(
      metadata.sigmaSquarePara, chainId)
    sigmaSquare.update(forest.residualsSquareSum)
    val halfNorm = new HalfNormSampler(metadata.tauPara, chainId)
    val weightedCount = df.map(_.weight).reduce(_ + _)
    val minMaxScale = metadata.inputSummarizer.minMaxScale
    val scale = math.pow(minMaxScale, 2) / weightedCount
    new Chain(forest, sigmaSquare, halfNorm, scale, parallelChainId)
  }
}
