package bart.configuration

trait Metadata extends Serializable {

  val inputSummarizer: InputSummarizer
  val categoryFeatureIndexes: Array[Int]
  val categoryFeatureArity: Array[Int]
  val maxBins: Int
  val maxArity: Int
  val chainCnt: Int
  val parallelChainCnt: Int
  val numBurn: Int
  val numSim: Int
  val numThin: Int

  val _lambda: Double
  val q: Double
  val nu: Double

  val minInstancesPerNode: Int
  val probGrow: Double
  val probPrune: Double
  val probChange: Double

  require(categoryFeatureIndexes.length == categoryFeatureArity.length, "categoryFeatureIndexes and" +
    "categoryFeatureArity should have the same length")
  val featureArity: Map[Int, Int] = categoryFeatureIndexes.zip(categoryFeatureArity).toMap

  require(probGrow + probPrune + probChange == 1.0,
    "grow, prune and change prob sum should be 1.0")
  def isCategorical(featureIndex: Int): Boolean = categoryFeatureIndexes.contains(featureIndex)
  def isContinuous(featureIndex: Int): Boolean = !categoryFeatureIndexes.contains(featureIndex)
}
