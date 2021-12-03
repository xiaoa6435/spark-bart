package bart.discretizer

import bart.Instance
import bart.configuration.{BARTMetadata, BCFMetadata, Metadata}
import org.apache.spark.rdd.RDD

/**
 * Internal representation of LabeledPoint for DecisionTree.
 * This bins feature values based on a subsampled of data as follows:
 * (a) Continuous features are binned into ranges.
 * (b) Unordered categorical features are binned based on subsets of feature values.
 * "Unordered categorical features" are categorical features with low arity used in
 * multiclass classification.
 * (c) Ordered categorical features are binned based on feature values.
 * "Ordered categorical features" are categorical features with high arity,
 * or any categorical feature used in regression or binary classification.
 *
 * @param label          Label from LabeledPoint
 * @param binnedFeatures Binned feature values.
 *                       Same length as LabeledPoint.features, but values are bin indices.
 */

class TreePoint(
  val binnedFeatures: Array[Int],
  val label: Double,
  val weight: Double,
  val resp: Array[Double],
  val chainCnt: Int
) extends Serializable {

  def getResidual(chainId: Int): Double = {
    resp(chainId)
  }

  def setResidual(value: Double, chainId: Int): Unit = {
    resp(chainId) = value
  }

  def getTmpResp(chainId: Int): Double = {
    resp(chainCnt + chainId)
  }

  def setTmpResp(value: Double, chainId: Int): Unit = {
    resp(chainCnt + chainId) = value
  }

  def getTmpWeight(chainId: Int): Double = {
    resp(2 * chainCnt + chainId)
  }

  def setTmpWeight(value: Double, chainId: Int): Unit = {
    resp(2 * chainCnt + chainId) = value
  }

  def getConResp(chainId: Int): Double = {
    resp(3 * chainCnt + chainId)
  }

  def setConResp(value: Double, chainId: Int): Unit = {
    resp(3 * chainCnt + chainId) = value
  }

  def getModResp(chainId: Int): Double = {
    resp(4 * chainCnt + chainId)
  }

  def setModResp(value: Double, chainId: Int): Unit = {
    resp(4 * chainCnt + chainId) = value
  }
}

object TreePoint {

  /**
   * Convert an input dataset into its TreePoint representation,
   * binning feature values in preparation for DecisionTree training.
   *
   * @param input    Input dataset.
   * @param splits   Splits for features, of size (numFeatures, numSplits).
   * @param metadata Learning and dataset para
   * @return TreePoint dataset representation
   */
  def convertToTreeRDD(
    input: RDD[Instance],
    splits: Array[Array[Split]],
    metadata: Metadata): RDD[TreePoint] = {
    // Construct arrays for featureArity for efficiency in the inner loop.
    val featureArity: Array[Int] = new Array[Int](metadata.inputSummarizer.numFeatures)
    var featureIndex = 0
    while (featureIndex < metadata.inputSummarizer.numFeatures) {
      featureArity(featureIndex) = metadata.featureArity.getOrElse(featureIndex, 0)
      featureIndex += 1
    }
    val thresholds: Array[Array[Double]] = featureArity.zipWithIndex.map { case (arity, idx) =>
      if (arity == 0) {
        splits(idx).map(_.asInstanceOf[ContinuousSplit].threshold)
      } else {
        Array.empty[Double]
      }
    }
    input.map { x =>
      TreePoint.labeledPointToTreePoint(x, thresholds, featureArity, metadata)
    }
  }

  /**
   * Convert one LabeledPoint into its TreePoint representation.
   *
   * @param thresholds   For each feature, split thresholds for continuous features,
   *                     empty for categorical features.
   * @param featureArity Array indexed by feature, with value 0 for continuous and numCategories
   *                     for categorical features.
   */
  private def labeledPointToTreePoint(
    labeledPoint: Instance,
    thresholds: Array[Array[Double]],
    featureArity: Array[Int],
    metadata: Metadata): TreePoint = {

    val numFeatures = labeledPoint.features.length
    val arr = new Array[Int](numFeatures)
    var featureIndex = 0
    while (featureIndex < numFeatures) {
      arr(featureIndex) =
        findBin(featureIndex, labeledPoint, featureArity(featureIndex), thresholds(featureIndex))
      featureIndex += 1
    }

    val label = labeledPoint.label
    val (resp, feat, init, residual) = metadata match {
      case m: BARTMetadata =>
        val scaleLabel = m.inputSummarizer.minMaxScaleLabel(label)
        val rawLabelMean = m.inputSummarizer.labelSummarizer.mean

        //val scaleLabelMean = m.inputSummarizer.scaleLabel(rawLabelMean)
//        val scaleLabelMean = 0.0
//        val residual = scaleLabel - scaleLabelMean
//        val initResp = scaleLabelMean / m.forestPara.numTrees
        val resp = Array.fill(m.chainCnt * 2)(0.0)
//        (resp, arr, initResp, residual)
        (resp, arr, 0.0, scaleLabel)
      case m: BCFMetadata =>
        val scaleLabel = metadata.inputSummarizer.normScaleLabel(label)
        val numFeatures = m.inputSummarizer.numFeatures
        val resp = Array.fill(m.chainCnt * 5)(0.0)
        val _arr = arr.slice(0, m.treatFeatIndex) ++
          arr.slice(m.treatFeatIndex + 1, numFeatures) ++
          Array(arr(m.treatFeatIndex))

        (resp, _arr, 0.0, scaleLabel)
    }

    val point = new TreePoint(feat, label, labeledPoint.weight, resp, metadata.chainCnt)
    Range(0, metadata.chainCnt).foreach(chainId => point.setTmpResp(init, chainId))
    Range(0, metadata.chainCnt).foreach(chainId => point.setResidual(residual, chainId))
    point
  }

  /**
   * Find discretized value for one (labeledPoint, feature).
   *
   * NOTE: We cannot use Bucketizer since it handles split thresholds differently than the old
   * (mllib) tree API.  We want to maintain the same behavior as the old tree API.
   *
   * @param featureArity 0 for continuous features; number of categories for categorical features.
   */
  private def findBin(
    featureIndex: Int,
    labeledPoint: Instance,
    featureArity: Int,
    thresholds: Array[Double]): Int = {
    val featureValue = labeledPoint.features(featureIndex)

    if (featureArity == 0) {
      val idx = java.util.Arrays.binarySearch(thresholds, featureValue)
      if (idx >= 0) {
        idx
      } else {
        -idx - 1
      }
    } else {
      // Categorical feature bins are indexed by feature values.
      if (featureValue < 0 || featureValue >= featureArity) {
        throw new IllegalArgumentException(
          s"DecisionTree given invalid data:" +
            s" Feature $featureIndex is categorical with values in {0,...,${featureArity - 1}," +
            s" but a data point gives it value $featureValue.\n" +
            "  Bad data point: " + labeledPoint.toString)
      }

      featureValue.toInt
    }
  }
}