package org.apache.spark.ml.regression

import bart.configuration.{BARTMetadata, InputSummarizer}
import bart.discretizer.{FindSplits, TreePoint}
import bart.sampler.Chain
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.{BayesAdditiveTreeModel, BayesAdditiveTreeModelReadWrite}
import org.apache.spark.ml.functions.checkNonNegativeWeight
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Column, DataFrame, Row}
import org.apache.spark.sql.functions.{lit, udf}
import org.apache.spark.sql.types.DoubleType
import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._
import org.apache.spark.ml.BayesAdditiveTreeRegressorParams
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions.col

/**
 * <a href="http://en.wikipedia.org/wiki/Decision_tree_learning">Decision tree</a>
 * learning algorithm for regression.
 * It supports both continuous and categorical features.
 */
class BayesAdditiveTreeRegressor (override val uid: String)
  extends Regressor[Vector, BayesAdditiveTreeRegressor, BayesAdditiveTreeRegressionModel]
    with BayesAdditiveTreeRegressorParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("bart"))

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

class BayesAdditiveTreeRegressionModel private[ml] (
  override val uid: String,
  override val multiForest: Seq[Seq[bart.tree.Node]],
  val minValue: Double,
  val minMaxScale: Double)
  extends RegressionModel[Vector, BayesAdditiveTreeRegressionModel]
    with BayesAdditiveTreeModel with BayesAdditiveTreeRegressorParams with MLWritable with Serializable {

  require(multiForest != null,
    "BayesAdditiveTreeRegressionModel given null rootNode, but it requires a non-null rootNode.")

  /**
   * Construct a bayes additive tree regression model.
   *
   * @param multiForest  multi sample of forest
   */
  private[ml] def this(multiForest: Seq[Seq[bart.tree.Node]], minValue: Double, minMaxScale: Double) =
    this(Identifiable.randomUID("bart"), multiForest, minValue, minMaxScale)

  override def predict(features: Vector): Double = {
    val raw = multiForest.map(f => f.map(t => t.predictImpl(features).mu).sum).sum / multiForest.length
    (raw + 0.5) * minMaxScale + minValue
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val outputSchema = transformSchema(dataset.schema, logging = true)

    var predictionColNames = Seq.empty[String]
    var predictionColumns = Seq.empty[Column]

    val bcastModel = dataset.sparkSession.sparkContext.broadcast(this)
    if ($(predictionCol).nonEmpty) {
      val predictUDF = udf { features: Vector => bcastModel.value.predict(features) }
      predictionColNames :+= $(predictionCol)
      predictionColumns :+= predictUDF(col($(featuresCol)))
        .as($(predictionCol), outputSchema($(predictionCol)).metadata)
    }

    if (predictionColNames.nonEmpty) {
      dataset.withColumns(predictionColNames, predictionColumns)
    } else {
      this.logWarning(s"$uid: BayesAdditiveTreeRegressionModel.transform() does nothing" +
        " because no output columns were set.")
      dataset.toDF()
    }
  }

  override def copy(extra: ParamMap): BayesAdditiveTreeRegressionModel = {
    copyValues(new BayesAdditiveTreeRegressionModel(uid, multiForest, minValue, minMaxScale), extra)
      .setParent(parent)
  }

  override def toString: String = {
    s"BayesAdditiveTreeRegressionModel: uid=$uid" +
      s"numFeatures=$numFeatures"
  }

  override def write: MLWriter =
    new BayesAdditiveTreeRegressionModel.BayesAdditiveTreeRegressionModelWriter(this)
}

object BayesAdditiveTreeRegressionModel extends MLReadable[BayesAdditiveTreeRegressionModel] {

  override def read: MLReader[BayesAdditiveTreeRegressionModel] =
    new BayesAdditiveTreeRegressionModelReader

  override def load(path: String): BayesAdditiveTreeRegressionModel = super.load(path)

  private[BayesAdditiveTreeRegressionModel]
  class BayesAdditiveTreeRegressionModelWriter(instance: BayesAdditiveTreeRegressionModel)
    extends MLWriter {

    override protected def saveImpl(path: String): Unit = {

      val extraMetadata: JObject = Map(
        "minValue" -> instance.minValue,
        "minMaxScale" -> instance.minMaxScale)
      DefaultParamsWriter.saveMetadata(instance, path, sc, Some(extraMetadata))
    }
  }

  private class BayesAdditiveTreeRegressionModelReader
    extends MLReader[BayesAdditiveTreeRegressionModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[BayesAdditiveTreeRegressionModel].getName

    override def load(path: String): BayesAdditiveTreeRegressionModel = {

      implicit val format: DefaultFormats.type = DefaultFormats
      val dataPath = new Path(path, "data").toString
      val (metadata: Metadata, multiForest: Seq[Seq[bart.tree.Node]]) =
        BayesAdditiveTreeModelReadWrite.loadImpl(dataPath, sparkSession, className)
      val minValue = (metadata.metadata \ "minValue").extract[Double]
      val minMaxScale = (metadata.metadata \ "minMaxScale").extract[Double]

      val model = new BayesAdditiveTreeRegressionModel(metadata.uid, multiForest, minValue, minMaxScale)
      metadata.getAndSetParams(model)
      model
    }
  }
}