package org.apache.spark.ml

import org.apache.spark.ml.classification.ProbabilisticClassifierParams
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.sql.types.{DataType, StructType}

/**
 * Parameters for Decision Tree-based algorithms.
 *
 * Note: Marked as private since this may be made public in the future.
 */
trait BayesAdditiveTreeParams extends PredictorParams
  with HasSeed with HasWeightCol {

  final val categoryFeatureIndexes: IntArrayParam = new IntArrayParam(this, name = "categoryFeatureIndexes", doc = "categoryFeatureIndexes")
  final def getCategoryFeatureIndexes: Array[Int] =  $(categoryFeatureIndexes)
  final def setCategoryFeatureIndexes(value: Array[Int]): this.type = set(categoryFeatureIndexes, value)

  final val categoryFeatureArity: IntArrayParam = new IntArrayParam(this, name = "categoryFeatureArity", doc = "categoryFeatureArity")
  final def getCategoryFeatureArity: Array[Int] =  $(categoryFeatureArity)
  final def setCategoryFeatureArity(value: Array[Int]): this.type = set(categoryFeatureArity, value)
  /**
   * Maximum number of bins used for discretizing continuous features and for choosing how to split
   * on features at each node.  More bins give higher granularity.
   * Must be at least 2 and at least number of categories in any categorical feature.
   * (default = 32)
   * @group param
   */
  final val maxBins: IntParam = new IntParam(this, "maxBins", "Max number of bins for" +
    " discretizing continuous features.  Must be at least 2 and at least number of categories" +
    " for any categorical feature.", ParamValidators.gtEq(2))

  /** @group getParam */
  final def getMaxBins: Int = $(maxBins)
  /** @group setParam */
  final def setMaxBins(value: Int): this.type = set(maxBins, value)

  final val maxArity: IntParam = new IntParam(this, "maxArity", "Max Arity",
    ParamValidators.gtEq(2))
  /** @group getParam */
  final def getMaxArity: Int = $(maxArity)
  /** @group setParam */
  final def setMaxArity(value: Int): this.type = set(maxArity, value)

  final val chainCnt: IntParam = new IntParam(this, "chainCnt", "chainCnt",
    ParamValidators.gtEq(0))
  /** @group getParam */
  final def getChainCnt: Int = $(chainCnt)
  /** @group setParam */
  final def setChainCnt(value: Int): this.type = set(chainCnt, value)

  final val parallelChainCnt: IntParam = new IntParam(this, "parallelChainCnt", "parallelChainCnt",
    ParamValidators.gtEq(0))
  /** @group getParam */
  final def getParallelChainCnt: Int = $(parallelChainCnt)
  /** @group setParam */
  final def setParallelChainCnt(value: Int): this.type = set(parallelChainCnt, value)

  /**
   * Minimum number of instances each child must have after split.
   * If a split causes the left or right child to have fewer than minInstancesPerNode,
   * the split will be discarded as invalid.
   * Must be at least 1.
   * (default = 1)
   * @group param
   */
  final val minInstancesPerNode: IntParam = new IntParam(this, "minInstancesPerNode", "Minimum" +
    " number of instances each child must have after split.  If a split causes the left or right" +
    " child to have fewer than minInstancesPerNode, the split will be discarded as invalid." +
    " Must be at least 1.", ParamValidators.gtEq(1))

  /** @group getParam */
  final def getMinInstancesPerNode: Int = $(minInstancesPerNode)

  /** @group setParam */
  final def setMinInstancesPerNode(value: Int): this.type = set(minInstancesPerNode, value)

  /**
   * Number of trees to train (at least 1).
   * (default = 200)
   * @group param
   */
  final val numTrees: IntParam =
    new IntParam(this, "numTrees", "Number of trees to train (at least 1)",
      ParamValidators.gtEq(1))

  /** @group getParam */
  final def getNumTrees: Int = $(numTrees)

  /** @group setParam */
  final def setNumTrees(value: Int): this.type = set(numTrees, value)

  final val numBurn: IntParam = new IntParam(this, "numBurn", "numBurn",
    ParamValidators.gtEq(1))
  /** @group getParam */
  final def getNumBurn: Int = $(numBurn)
  /** @group setParam */
  final def setNumBurn(value: Int): this.type = set(numBurn, value)

  final val numSim: IntParam = new IntParam(this, "numSim", "numSim",
    ParamValidators.gtEq(1))
  /** @group getParam */
  final def getNumSim: Int = $(numSim)
  /** @group setParam */
  final def setNumSim(value: Int): this.type = set(numSim, value)

  final val numThin: IntParam = new IntParam(this, "numThin", "numThin",
    ParamValidators.gtEq(1))
  /** @group getParam */
  final def getNumThin: Int = $(numThin)
  /** @group setParam */
  final def setNumThin(value: Int): this.type = set(numThin, value)

  final val probGrow: DoubleParam = new DoubleParam(this, "probGrow", "probGrow",
    ParamValidators.gtEq(0.0))
  /** @group getParam */
  final def getProbGrow: Double = $(probGrow)
  /** @group setParam */
  final def setProbGrow(value: Double): this.type = set(probGrow, value)

  final val probPrune: DoubleParam = new DoubleParam(this, "probPrune", "probPrune",
    ParamValidators.gtEq(0.0))
  /** @group getParam */
  final def getProbPrune: Double = $(probPrune)
  /** @group setParam */
  final def setProbPrune(value: Double): this.type = set(probPrune, value)

  final val probChange: DoubleParam = new DoubleParam(this, "probChange", "probChange",
    ParamValidators.gtEq(0.0))
  /** @group getParam */
  final def getProbChange: Double = $(probChange)
  /** @group setParam */
  final def setProbChange(value: Double): this.type = set(probChange, value)

  final val alpha: DoubleParam = new DoubleParam(this, "alpha", "alpha",
    ParamValidators.gtEq(0.0))
  /** @group getParam */
  final def getAlpha: Double = $(alpha)
  /** @group setParam */
  final def setAlpha(value: Double): this.type = set(alpha, value)

  final val beta: DoubleParam = new DoubleParam(this, "beta", "beta",
    ParamValidators.gtEq(0.0))
  /** @group getParam */
  final def getBeta: Double = $(beta)
  /** @group setParam */
  final def setBeta(value: Double): this.type = set(beta, value)

  final val sd: DoubleParam = new DoubleParam(this, "sd", "sd",
    ParamValidators.gtEq(0.0))
  /** @group getParam */
  final def getSd: Double = $(sd)
  /** @group setParam */
  final def setSd(value: Double): this.type = set(sd, value)

  final val lambdaRaw: DoubleParam = new DoubleParam(this, "lambdaRaw", "lambdaRaw",
    ParamValidators.gtEq(0.0))
  /** @group getParam */
  final def getLambdaRaw: Double = $(lambdaRaw)
  /** @group setParam */
  final def setLambdaRaw(value: Double): this.type = set(lambdaRaw, value)

  final val q: DoubleParam = new DoubleParam(this, "q", "q",
    ParamValidators.gtEq(0.0))
  /** @group getParam */
  final def getQ: Double = $(q)
  /** @group setParam */
  final def setQ(value: Double): this.type = set(q, value)

  final val nu: DoubleParam = new DoubleParam(this, "nu", "nu",
    ParamValidators.gtEq(0.0))
  /** @group getParam */
  final def getNu: Double = $(nu)
  /** @group setParam */
  final def setNu(value: Double): this.type = set(nu, value)

  setDefault(
    categoryFeatureIndexes -> Array.emptyIntArray,
    categoryFeatureArity -> Array.emptyIntArray,
    maxBins -> 32, maxArity -> 5,
    chainCnt -> 1, parallelChainCnt -> 1, minInstancesPerNode -> 1,
    numTrees -> 200, numBurn -> 1000, numSim -> 100, numThin -> 5,
    probGrow -> 2.5 / 9.0, probPrune -> 2.5 / 9.0, probChange -> 4.0 / 9.0,
    alpha -> 0.95, beta -> 2.0, sd -> 0.25,
    lambdaRaw -> 0.0, q -> 0.9, nu -> 3.0
//    maxMemoryInMB -> 256, cacheNodeIds -> false, checkpointInterval -> 10
  )

}

/**
 * Parameters for Decision Tree-based classification algorithms.
 */

trait BayesAdditiveTreeClassifierParams
  extends BayesAdditiveTreeParams with ProbabilisticClassifierParams {

  override protected def validateAndTransformSchema(
    schema: StructType,
    fitting: Boolean,
    featuresDataType: DataType): StructType = {
    val outputSchema = super.validateAndTransformSchema(schema, fitting, featuresDataType)
    outputSchema
  }
}

/**
 * Parameters for Decision Tree-based regression algorithms.
 */
trait BayesAdditiveTreeRegressorParams extends BayesAdditiveTreeParams {
  override protected def validateAndTransformSchema(
    schema: StructType,
    fitting: Boolean,
    featuresDataType: DataType): StructType = {
    val outputSchema = super.validateAndTransformSchema(schema, fitting, featuresDataType)
    outputSchema
  }
}
