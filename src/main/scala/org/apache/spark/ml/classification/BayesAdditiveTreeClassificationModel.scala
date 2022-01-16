package org.apache.spark.ml.classification

import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.{BayesAdditiveTreeModel, BayesAdditiveTreeModelReadWrite}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.apache.spark.ml.util.{DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter}
import org.apache.spark.sql.{Column, DataFrame, Dataset}
import org.apache.spark.sql.functions.{col, udf}
import org.json4s.JsonDSL._
import org.json4s.{DefaultFormats, JObject}

class BayesAdditiveTreeClassificationModel private[ml] (
  override val uid: String,
  override val multiForest: Seq[Seq[bart.tree.Node]],
  val minValue: Double,
  val minMaxScale: Double)
  extends ClassificationModel[Vector, BayesAdditiveTreeClassificationModel]
    with BayesAdditiveTreeModel with MLWritable with Serializable {

  require(multiForest != null,
    "BayesAdditiveTreeClassificationModel given null rootNode, but it requires a non-null rootNode.")

  /**
   * Construct a bayes additive tree Classification model.
   *
   * @param multiForest  multi sample of forest
   */
  private[ml] def this(multiForest: Seq[Seq[bart.tree.Node]], minValue: Double, minMaxScale: Double) =
    this(Identifiable.randomUID("bart"), multiForest, minValue, minMaxScale)

  override def predict(features: Vector): Double = {
    val raw = predictRaw(features).toArray.head
    val g = new NormalDistribution
    g.cumulativeProbability(raw)
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
      this.logWarning(s"$uid: BayesAdditiveTreeClassificationModel.transform() does nothing" +
        " because no output columns were set.")
      dataset.toDF()
    }
  }

  override def copy(extra: ParamMap): BayesAdditiveTreeClassificationModel = {
    copyValues(new BayesAdditiveTreeClassificationModel(uid, multiForest, minValue, minMaxScale), extra)
      .setParent(parent)
  }

  override def toString: String = {
    s"BayesAdditiveTreeClassificationModel: uid=$uid" +
      s"numFeatures=$numFeatures"
  }

  override def write: MLWriter =
    new BayesAdditiveTreeClassificationModel.BayesAdditiveTreeClassificationModelWriter(this)

  override def numClasses: Int = 2

  override def predictRaw(features: Vector): Vector = {
    val raw = multiForest.map(f => f.map(t => t.predictImpl(features).mu).sum).sum / multiForest.length
    Vectors.dense(Array(raw, -raw))
  }
}

object BayesAdditiveTreeClassificationModel extends MLReadable[BayesAdditiveTreeClassificationModel] {

  override def read: MLReader[BayesAdditiveTreeClassificationModel] =
    new BayesAdditiveTreeClassificationModelReader

  override def load(path: String): BayesAdditiveTreeClassificationModel = super.load(path)

  private[BayesAdditiveTreeClassificationModel]
  class BayesAdditiveTreeClassificationModelWriter(instance: BayesAdditiveTreeClassificationModel)
    extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      val extraMetadata: JObject = Map(
        "minValue" -> instance.minValue,
        "minMaxScale" -> instance.minMaxScale)
      DefaultParamsWriter.saveMetadata(instance, path, sc, Some(extraMetadata))
    }
  }

  private class BayesAdditiveTreeClassificationModelReader
    extends MLReader[BayesAdditiveTreeClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[BayesAdditiveTreeClassificationModel].getName

    override def load(path: String): BayesAdditiveTreeClassificationModel = {
      implicit val format: DefaultFormats.type = DefaultFormats
      val dataPath = new Path(path, "data").toString
      val (metadata: Metadata, multiForest: Seq[Seq[bart.tree.Node]]) =
        BayesAdditiveTreeModelReadWrite.loadImpl(dataPath, sparkSession, className)
      val minValue = (metadata.metadata \ "minValue").extract[Double]
      val minMaxScale = (metadata.metadata \ "minMaxScale").extract[Double]
      val model = new BayesAdditiveTreeClassificationModel(metadata.uid, multiForest, minValue, minMaxScale)
      metadata.getAndSetParams(model)
      model
    }
  }
}