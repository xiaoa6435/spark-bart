package org.apache.spark.ml.regression

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.apache.spark.ml.util._
import org.apache.spark.ml.{BayesAdditiveTreeModel, BayesAdditiveTreeModelReadWrite}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{Column, DataFrame, Dataset}
import org.json4s.JsonDSL._
import org.json4s.{DefaultFormats, JObject}

class BayesAdditiveTreeRegressionModel private[ml] (
  override val uid: String,
  override val multiForest: Seq[Seq[bart.tree.Node]],
  val minValue: Double,
  val minMaxScale: Double)
  extends RegressionModel[Vector, BayesAdditiveTreeRegressionModel]
    with BayesAdditiveTreeModel with MLWritable with Serializable {

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