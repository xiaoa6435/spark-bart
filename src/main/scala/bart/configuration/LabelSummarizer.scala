package bart.configuration

class LabelSummarizer(isFrequencyWeight: Boolean = true) extends Serializable {
  private var currMean: Double = 0.0
  private var currM2n: Double = 0.0
  private var currMax: Double = Double.MinValue
  private var currMin: Double = Double.MaxValue
  private var cnt: Long = 0
  private var weightSum: Double = 0.0
  private var weightSquareSum: Double = 0.0

  /**
   * Add a new sample to this summarizer, and update the statistical summary.
   *
   * @return This LabelSummarizer object.
   */
  def add(label: Double, weight: Double): this.type = {

    require(weight >= 0.0, s"sample weight, $weight has to be >= 0.0")

    if (weight == 0.0) return this

    cnt += 1
    weightSum += weight
    weightSquareSum += weight * weight
    currMax = currMax.max(label)
    currMin = currMin.min(label)

    val diff = label - currMean
    currMean += weight * diff / weightSum
    currM2n += weight * (label - currMean) * diff

    this
  }

  /**
   * Merge another LabelSummarizer, and update the statistical summary.
   * (Note that it's in place merging; as a result, `this` object will be modified.)
   *
   * @param other The other LabelSummarizer to be merged.
   * @return This LabelSummarizer  object.
   */
  def merge(other: LabelSummarizer): this.type = {

    cnt += other.cnt
    weightSum += other.weightSum
    if (weightSum == 0.0) return this

    weightSquareSum += other.weightSquareSum
    currMax = currMax.max(other.currMax)
    currMin = currMin.min(other.currMin)

    val deltaMean = other.currMean - currMean
    currMean += deltaMean * other.weightSum / weightSum
    val _w = (weightSum - other.weightSum) * other.weightSum / weightSum
    currM2n += other.currM2n + math.pow(deltaMean, 2) * _w
    this
  }

  /**
   * Sample mean of each dimension.
   *
   */
  def mean: Double = currMean

  /**
   * Unbiased estimate of sample variance of each dimension.
   *
   */
  def variance: Double = {
    require(weightSum > 0, s"Nothing has been added to this summarizer.")
    val correction = if (isFrequencyWeight) 1 else weightSquareSum / weightSum
    val denominator = weightSum - correction
    (currM2n / denominator).max(0.0)
  }

  /**
   * Sample size.
   *
   */
  def count: Long = cnt

  /**
   * Sum of weights.
   */
  def weightedCount: Double = weightSum

  /**
   * Maximum value of each dimension.
   *
   */
  def max: Double = currMax

  /**
   * Minimum value of each dimension.
   *
   */
  def min: Double = currMin
}