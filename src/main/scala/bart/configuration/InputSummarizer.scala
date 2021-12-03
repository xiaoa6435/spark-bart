package bart.configuration

import bart.Instance
import org.apache.spark.rdd.RDD

class InputSummarizer(
  val numFeatures: Int,
  val numPoints: Long,
  val labelSummarizer: LabelSummarizer) extends Serializable {

  val minMaxScale: Double = labelSummarizer.max - labelSummarizer.min

  def scaleLabel(label: Double): Double = (label - labelSummarizer.min) / minMaxScale - 0.5
  def unScaleLabel(scaledLabel: Double): Double = {
    (scaledLabel + 0.5) * minMaxScale + labelSummarizer.min
  }

  def minMaxScaleLabel(label: Double): Double = (label - labelSummarizer.min) / minMaxScale - 0.5
  def normScaleLabel(label: Double): Double = (label - labelSummarizer.mean) / math.sqrt(labelSummarizer.variance)

  def unMinMaxScaleLabel(scaledLabel: Double): Double = {
    (scaledLabel + 0.5) * minMaxScale + labelSummarizer.min
  }

  def unNormScaleLabel(scaledLabel: Double): Double = {
    scaledLabel * math.sqrt(labelSummarizer.variance) + labelSummarizer.mean
  }

}

object InputSummarizer {
  def calculate(input: RDD[Instance], isFrequencyWeight: Boolean = true): InputSummarizer = {

    val numFeatures: Int = input.map(_.features.length).take(1).headOption.getOrElse {
      throw new IllegalArgumentException(s"DecisionTree requires size of input RDD > 0, " +
        s"but was given by empty one.")
    }
    require(numFeatures > 0, s"DecisionTree requires number of features > 0, " +
      s"but was given an empty features vector")
    val numPoints: Long = input.count

    val labelSummarizer = {
      input.treeAggregate(new LabelSummarizer(isFrequencyWeight))(
        seqOp = { case (c, p) => c.add(p.label, p.weight) },
        combOp = { case (c1, c2) => c1.merge(c2) },
        depth = 2
      )
    }
    new InputSummarizer(numFeatures, numPoints, labelSummarizer)
  }
}