package org.apache.spark.ml.regression

import bart.configuration.{BARTMetadata, InputSummarizer}
import bart.discretizer.{FindSplits, TreePoint}
import bart.sampler.Chain
import org.apache.spark.ml.BayesAdditiveTreeParams
import org.apache.spark.ml.functions.checkNonNegativeWeight
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.types.DoubleType

//trait BayesAdditiveTreeRegressorParams extends BayesAdditiveTreeParams
/**
 * <a href="http://en.wikipedia.org/wiki/Decision_tree_learning">Decision tree</a>
 * learning algorithm for regression.
 * It supports both continuous and categorical features.
 */
class BayesAdditiveTreeRegressor (override val uid: String)
  extends Regressor[Vector, BayesAdditiveTreeRegressor, BayesAdditiveTreeRegressionModel]
    with BayesAdditiveTreeParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("bart"))

  /** @group setParam */
  def setSd(value: Double): this.type = set(sd, value)
  setDefault(sd -> 0.25)

  def _extractInstances(dataset: Dataset[_]): RDD[bart.Instance] = {
    val w = this match {
      case p: HasWeightCol =>
        if (isDefined(p.weightCol) && $(p.weightCol).nonEmpty) {
          checkNonNegativeWeight(col($(p.weightCol)).cast(DoubleType))
        } else {
          lit(1.0)
        }
    }

    dataset.select(col($(labelCol)).cast(DoubleType), w, col($(featuresCol))).rdd.map {
      case Row(label: Double, weight: Double, features: Vector) =>
        bart.Instance(label, features.toArray, weight)
    }
  }
  override protected def train(
    dataset: Dataset[_]): BayesAdditiveTreeRegressionModel = instrumented { instr =>

    val instances = _extractInstances(dataset)
    instr.logPipelineStage(this)
    instr.logDataset(instances)
    instr.logParams(this, params: _*)

    val inputSummarizer = InputSummarizer.calculate(instances)
    val metadata = new BARTMetadata(
      inputSummarizer, $(categoryFeatureIndexes), $(categoryFeatureArity),
      $(maxBins), $(maxArity), $(chainCnt), $(parallelChainCnt),
      $(numBurn), $(numSim), $(numThin), $(numTrees), $(alpha), $(beta), $(sd),
      $(minInstancesPerNode), $(probGrow), $(probPrune), $(probChange),
      $(lambdaRaw), $(q), $(nu)
    )
    val splits = FindSplits.findSplits(instances, metadata, seed = $(seed))
    val treeInput = TreePoint.convertToTreeRDD(instances, splits, metadata)

    val resForest = treeInput.
      repartition(metadata.parallelChainCnt).
      mapPartitionsWithIndex{case (parallelChainId, iter)=>
        val dfSeq = new bart.SeqArr(iter.toArray)
        val chain = Chain(dfSeq, metadata, parallelChainId)
        chain.run(metadata.numBurn)
        val iResForest = chain.run(
          metadata.numSim, metadata.numThin, splits)
        iResForest.iterator
      }.collect().map(_.toSeq).toSeq
    val minValue = inputSummarizer.labelSummarizer.min
    val minMaxScale = inputSummarizer.minMaxScale
    new BayesAdditiveTreeRegressionModel(resForest, minValue, minMaxScale)
  }

  override def copy(extra: ParamMap): BayesAdditiveTreeRegressor = defaultCopy(extra)
}

