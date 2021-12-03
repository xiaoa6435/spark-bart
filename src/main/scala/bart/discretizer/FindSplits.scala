package bart.discretizer

import bart.Instance
import bart.configuration.Metadata
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD

import scala.collection.mutable

object FindSplits extends Logging with Serializable {
  /**
   * Returns splits for decision tree calculation.
   * Continuous and categorical features are handled differently.
   *
   * Continuous features:
   * For each feature, there are numBins - 1 possible splits representing the possible binary
   * decisions at each node in the tree.
   * This finds locations (feature values) for splits using a subsample of the data.
   *
   * Categorical features:
   * For each feature, there is 1 bin per split.
   * Splits and bins are handled in 2 ways:
   * (a) "unordered features"
   * For multiclass classification with a low-arity feature
   * (i.e., if isMulticlass && isSpaceSufficientForAllCategoricalSplits),
   * the feature is split based on subsets of categories.
   * (b) "ordered features"
   * For regression and binary classification,
   * and for multiclass classification with a high-arity feature,
   * there is one bin per category.
   *
   * @param input    Training data: RDD of [[Instance]]
   * @param metadata Learning and dataset para
   * @param seed     random seed
   * @return Splits, an Array of [[Split]]
   *         of size (numFeatures, numSplits)
   */
  def findSplits(
    input: RDD[Instance],
    metadata: Metadata,
    seed: Long): Array[Array[Split]] = {

    val numFeatures = metadata.inputSummarizer.numFeatures

    // Sample the input only if there are continuous features.
    val continuousFeatures = Range(0, numFeatures).filter(metadata.isContinuous)
    val sampledInput = if (continuousFeatures.nonEmpty) {
      val fraction = samplesFractionForFindSplits(metadata)
      logDebug("fraction of data used for calculating quantiles = " + fraction)
      input.sample(withReplacement = false, fraction, new XORShiftRandom(seed).nextInt())
    } else {
      input.sparkContext.emptyRDD[Instance]
    }

    findSplitsBySorting(sampledInput, metadata, continuousFeatures)

  }

  private def findSplitsBySorting(
    input: RDD[Instance],
    metadata: Metadata,
    continuousFeatures: IndexedSeq[Int]): Array[Array[Split]] = {

    val continuousSplits: scala.collection.Map[Int, Array[Split]] = {
      // reduce the parallelism for split computations when there are less
      // continuous features than input partitions. this prevents tasks from
      // being spun up that will definitely do no work.
      val numPartitions = math.min(continuousFeatures.length, input.partitions.length)

      input
        .flatMap { point =>
          continuousFeatures.map(idx => (idx, point.features(idx))).filter(_._2 != 0.0)
        }.groupByKey(numPartitions)
        .map { case (idx, samples) =>

          val thresholds = findSplitsForContinuousFeature(samples, metadata, idx)
          val splits: Array[Split] = thresholds.zipWithIndex.
            map { case (thresh, binnedThresh) => new ContinuousSplit(idx, thresh, binnedThresh) }
          logDebug(s"featureIndex = $idx, numSplits = ${splits.length}")
          (idx, splits)

        }.collectAsMap()
    }

    val numFeatures = metadata.inputSummarizer.numFeatures
    val splits: Array[Array[Split]] = Array.tabulate(numFeatures) {
      case i if metadata.isContinuous(i) =>
        // some features may contain only zero, so continuousSplits will not have a record
        val split = continuousSplits.getOrElse(i, Array.empty[Split])
        split
      case i if metadata.isCategorical(i) =>
        Array.empty[Split]
    }
    splits
  }


  /**
   * Nested method to extract list of eligible categories given an index. It extracts the
   * position of ones in a binary representation of the input. If binary
   * representation of an number is 01101 (13), the output list should (3.0, 2.0,
   * 0.0). The maxFeatureValue depict the number of rightmost digits that will be tested for ones.
   */
  def extractMultiClassCategories(
    input: Int,
    maxFeatureValue: Int): List[Int] = {
    var categories = List[Int]()
    var j = 0
    var bitShiftedInput = input
    while (j < maxFeatureValue) {
      if (bitShiftedInput % 2 != 0) {
        // updating the list of categories.
        categories = j :: categories
      }
      // Right shift by one
      bitShiftedInput = bitShiftedInput >> 1
      j += 1
    }
    categories
  }

  /**
   * Find splits for a continuous feature
   * NOTE: Returned number of splits is set based on `featureSamples` and
   * could be different from the specified `numSplits`.
   * The `numSplits` attribute in the `Metadata` class will be set accordingly.
   *
   * @param featureSamples feature values of each sample
   * @param metadata       decision tree para
   *                       NOTE: `para.numbins` will be changed accordingly
   *                       if there are not enough splits to be found
   * @param featureIndex   feature index to find splits
   * @return array of split thresholds
   */
  def findSplitsForContinuousFeature(
    featureSamples: Iterable[Double],
    metadata: Metadata,
    featureIndex: Int): Array[Double] = {

    require(metadata.isContinuous(featureIndex),
      "findSplitsForContinuousFeature can only be used to find splits for a continuous feature.")

    val splits: Array[Double] = if (featureSamples.isEmpty) {
      Array.empty[Double]
    } else {
      //val numSplits = metadata.numSplits(featureIndex)
      val numSplits = metadata.maxBins
      // get count for each distinct value except zero value
      val partNumSamples = featureSamples.size
      val partValueCountMap = scala.collection.mutable.Map[Double, Int]()
      featureSamples.foreach { x =>
        partValueCountMap(x) = partValueCountMap.getOrElse(x, 0) + 1
      }

      // Calculate the expected number of samples for finding splits
      val numPoints = metadata.inputSummarizer.numPoints
      val numSamples = (samplesFractionForFindSplits(metadata) * numPoints).toInt
      // add expected zero value count and get complete statistics
      val valueCountMap: Map[Double, Int] = if (numSamples - partNumSamples > 0) {
        partValueCountMap.toMap + (0.0 -> (numSamples - partNumSamples))
      } else {
        partValueCountMap.toMap
      }

      // sort distinct values
      val valueCounts = valueCountMap.toSeq.sortBy(_._1).toArray

      val possibleSplits = valueCounts.length - 1
      if (possibleSplits == 0) {
        // constant feature
        Array.empty[Double]
      } else if (possibleSplits <= numSplits) {
        // if possible splits is not enough or just enough, just return all possible splits
        (1 to possibleSplits)
          .map(index => (valueCounts(index - 1)._1 + valueCounts(index)._1) / 2.0)
          .toArray
      } else {
        // stride between splits
        val stride: Double = numSamples.toDouble / (numSplits + 1)
        logDebug("stride = " + stride)

        // iterate `valueCount` to find splits
        val splitsBuilder = mutable.ArrayBuilder.make[Double]
        var index = 1
        // currentCount: sum of counts of values that have been visited
        var currentCount = valueCounts(0)._2
        // targetCount: target value for `currentCount`.
        // If `currentCount` is closest value to `targetCount`,
        // then current value is a split threshold.
        // After finding a split threshold, `targetCount` is added by stride.
        var targetCount = stride
        while (index < valueCounts.length) {
          val previousCount = currentCount
          currentCount += valueCounts(index)._2
          val previousGap = math.abs(previousCount - targetCount)
          val currentGap = math.abs(currentCount - targetCount)
          // If adding count of current value to currentCount
          // makes the gap between currentCount and targetCount smaller,
          // previous value is a split threshold.
          if (previousGap < currentGap) {
            splitsBuilder += (valueCounts(index - 1)._1 + valueCounts(index)._1) / 2.0
            targetCount += stride
          }
          index += 1
        }

        splitsBuilder.result()
      }
    }
    splits
  }

  /**
   * Calculate the subsample fraction for finding splits
   *
   * @param metadata decision tree para
   * @return subsample fraction
   */
  private def samplesFractionForFindSplits(
    metadata: Metadata): Double = {
    // Calculate the number of samples for approximate quantile calculation.
    val requiredSamples = math.max(metadata.maxBins * metadata.maxBins, 10000)
    if (requiredSamples < metadata.inputSummarizer.numPoints) {
      requiredSamples.toDouble / metadata.inputSummarizer.numPoints
    } else {
      1.0
    }
  }
}
