package org.apache.spark.ml

import bart.discretizer.{CategoricalSplit, ContinuousSplit, Split}

/**
 * Info for a [[org.apache.spark.ml.tree.Split]]
 *
 * @param featureIndex  Index of feature split on
 * @param leftCategoriesOrThreshold  For categorical feature, set of leftCategories.
 *                                   For continuous feature, threshold.
 * @param numCategories  For categorical feature, number of categories.
 *                       For continuous feature, -1.
 */
case class SplitData(
  featureIndex: Int,
  leftCategoriesOrThreshold: Array[Double],
  numCategories: Int) {

  def getSplit: Split = {
    if (numCategories != -1) {
      new CategoricalSplit(featureIndex, leftCategoriesOrThreshold.map(_.toInt), numCategories)
    } else {
      assert(leftCategoriesOrThreshold.length == 1, s"BayesAdditiveTree split data expected" +
        s" 1 threshold for ContinuousSplit, but found thresholds: " +
        leftCategoriesOrThreshold.mkString(", "))
      new ContinuousSplit(featureIndex, leftCategoriesOrThreshold(0), -1)
    }
  }
}

object SplitData {
  def apply(split: Split): SplitData = split match {
    case s: CategoricalSplit =>
      SplitData(s.featureIndex, s.leftCategories.map(_.toDouble), s.numCategories)
    case s: ContinuousSplit =>
      SplitData(s.featureIndex, Array(s.threshold), -1)
  }
}
