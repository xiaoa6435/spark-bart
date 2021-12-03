package bart.discretizer

import java.util.Objects

import org.apache.spark.ml.linalg.Vector

/**
 * Interface for a "Split," which specifies a test made at a decision tree node
 * to choose the left or right path.
 */
sealed trait Split extends Serializable {

  /** Index of feature which this split tests */
  def featureIndex: Int

  /**
   * Return true (split to left) or false (split to right).
   *
   * @param features Vector of features (original values, not binned).
   */
  def shouldGoLeft(features: Vector): Boolean

  def shouldGoLeft(binnedFeatures: Array[Int]): Boolean
  /**
   * Return true (split to left) or false (split to right).
   *
   * @param binnedFeature Binned feature value.
   */
  def shouldGoLeft(binnedFeature: Int): Boolean
}

/**
 * Split which tests a categorical feature.
 *
 * @param featureIndex    Index of the feature to test
 * @param _leftCategories If the feature value is in this set of categories, then the split goes
 *                        left. Otherwise, it goes right.
 * @param numCategories   Number of categories for this feature.
 */
class CategoricalSplit(
  override val featureIndex: Int,
  _leftCategories: Array[Int],
  val numCategories: Int) extends Split {

  require(_leftCategories.forall(cat => 0 <= cat && cat < numCategories),
    "Invalid leftCategories" +
      s" (should be in range [0, $numCategories)): ${_leftCategories.mkString(",")}")

  /**
   * If true, then "categories" is the set of categories for splitting to the left, and vice versa.
   */
  private val isLeft: Boolean = _leftCategories.length <= numCategories / 2

  /** Set of categories determining the splitting rule, along with [[isLeft]]. */
  //private val categories: Set[Double] = {
  val categories: Set[Int] = {
    if (isLeft) {
      _leftCategories.toSet
    } else {
      setComplement(_leftCategories.toSet)
    }
  }

  override def shouldGoLeft(features: Vector): Boolean = {
    if (isLeft) {
      categories.contains(features(featureIndex).toInt)
    } else {
      !categories.contains(features(featureIndex).toInt)
    }
  }

  override def shouldGoLeft(binnedFeatures: Array[Int]): Boolean = {
    if (isLeft) {
      categories.contains(binnedFeatures(featureIndex))
    } else {
      !categories.contains(binnedFeatures(featureIndex))
    }
  }

  override def shouldGoLeft(binnedFeature: Int): Boolean = {
    if (isLeft) {
      categories.contains(binnedFeature)
    } else {
      !categories.contains(binnedFeature)
    }
  }

  override def hashCode(): Int = {
    val state = Seq(featureIndex, isLeft, categories)
    state.map(Objects.hashCode).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def equals(o: Any): Boolean = o match {
    case other: CategoricalSplit => featureIndex == other.featureIndex &&
      isLeft == other.isLeft && categories == other.categories
    case _ => false
  }

  /** Get sorted categories which split to the left */
  def leftCategories: Array[Int] = {
    val cats = if (isLeft) categories else setComplement(categories)
    cats.toArray.sorted
  }

  /** Get sorted categories which split to the right */
  def rightCategories: Array[Int] = {
    val cats = if (isLeft) setComplement(categories) else categories
    cats.toArray.sorted
  }

  /** [0, numCategories) \ cats */
  private def setComplement(cats: Set[Int]): Set[Int] = {
    Range(0, numCategories).filter(cat => !cats.contains(cat)).toSet
  }

  override def toString: String = {
    val leftCategoriesStr = leftCategories.mkString("{", ",", "}")
    s"feature: $featureIndex, leftCategories: $leftCategoriesStr"
  }
}

/**
 * Split which tests a continuous feature.
 *
 * @param featureIndex Index of the feature to test
 * @param threshold    If the feature value is less than or equal to this threshold, then the
 *                     split goes left. Otherwise, it goes right.
 */
class ContinuousSplit(
  override val featureIndex: Int,
  val threshold: Double,
  val binnedThreshold: Int) extends Split {

  override def shouldGoLeft(features: Vector): Boolean = {
    features(featureIndex) <= threshold
  }

  override def shouldGoLeft(binnedFeatures: Array[Int]): Boolean = {
    binnedFeatures(featureIndex) <= binnedThreshold
  }

  override def shouldGoLeft(binnedFeature: Int): Boolean = {
    binnedFeature <= binnedThreshold
  }

  override def equals(o: Any): Boolean = {
    o match {
      case other: ContinuousSplit =>
        featureIndex == other.featureIndex && threshold == other.threshold &&
          binnedThreshold == other.binnedThreshold
      case _ =>
        false
    }
  }

  override def hashCode(): Int = {
    val state = Seq(featureIndex, threshold, binnedThreshold)
    state.map(Objects.hashCode).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def toString: String = s"feature: $featureIndex, " +
    s"""threshold: ${threshold.formatted("%.3f")}, binnedThreshold: $binnedThreshold"""
}
