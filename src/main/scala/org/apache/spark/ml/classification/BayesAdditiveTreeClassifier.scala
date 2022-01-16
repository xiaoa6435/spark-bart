package org.apache.spark.ml.classification

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

/**
 * <a href="http://en.wikipedia.org/wiki/Decision_tree_learning">Decision tree</a>
 * learning algorithm for Classification.
 * It supports both continuous and categorical features.
 */

class BayesAdditiveTreeClassifier (override val uid: String)
  extends Classifier[Vector, BayesAdditiveTreeClassifier, BayesAdditiveTreeClassificationModel]
    with BayesAdditiveTreeParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("bact"))

  /** @group setParam */
  def setSd(value: Double): this.type = set(sd, value)
  setDefault(sd -> 1.50)

  def _extractInstances(dataset: Dataset[_], numClasses: Int = 2): RDD[bart.Instance] = {
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
        require(label.toLong == label && label >= 0 && label < numClasses, s"Classifier was given" +
          s" dataset with invalid label $label. Labels must be integers in range" +
          s" [0, $numClasses).")
        bart.Instance(label, features.toArray, weight)
    }
  }

  //override protected def train(
  override def train(
    dataset: Dataset[_]): BayesAdditiveTreeClassificationModel = instrumented { instr =>

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
        val chain = Chain(dfSeq, metadata, parallelChainId, isClassifier = true)
        chain.run(metadata.numBurn)
        val iResForest = chain.run(
          metadata.numSim, metadata.numThin, splits)
        iResForest.iterator
      }.collect().map(_.toSeq).toSeq
    val minValue = inputSummarizer.labelSummarizer.min
    val minMaxScale = inputSummarizer.minMaxScale
    new BayesAdditiveTreeClassificationModel(resForest, minValue, minMaxScale)
  }
  override def copy(extra: ParamMap): BayesAdditiveTreeClassifier = defaultCopy(extra)
}

