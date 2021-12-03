package bart.tree

import bart.discretizer.{CategoricalSplit, ContinuousSplit, Split}
import org.apache.spark.ml.linalg.Vector

/**
 * Decision tree node interface.
 */
sealed abstract class Node extends Serializable {

  /** Recursive prediction helper method */
  def predictImpl(features: Vector): LeafNode

  /**
   * Recursive print function.
   *
   * @param indentFactor The number of spaces to add to each level of indentation.
   */
  def subtreeToString(indentFactor: Int = 0): String

}

/**
 * Decision tree leaf node.
 *
 * @param mu Prediction this node makes
 */
class LeafNode(val mu: Double) extends Node {

  override def toString: String = s"mu: $mu\n"

  override def predictImpl(features: Vector): LeafNode = this

  override def subtreeToString(indentFactor: Int = 0): String = {
    val prefix: String = " " * indentFactor
    prefix + s"mu: $mu\n"
  }
}

/**
 * Internal Decision Tree node.
 *
 * @param leftChild  Left-hand child node
 * @param rightChild Right-hand child node
 * @param split      Information about the test used to split to the left or right child.
 */
class InternalNode(
  val leftChild: Node,
  val rightChild: Node,
  val split: Split) extends Node {

  override def toString: String = {
    s"InternalNode($split)"
  }

  override def predictImpl(features: Vector): LeafNode = {
    if (split.shouldGoLeft(features)) {
      leftChild.predictImpl(features)
    } else {
      rightChild.predictImpl(features)
    }
  }

  override def subtreeToString(indentFactor: Int = 0): String = {
    val prefix: String = " " * indentFactor
    prefix + s"If (${InternalNode.splitToString(split, left = true)})\n" +
      leftChild.subtreeToString(indentFactor + 1) +
      prefix + s"Else (${InternalNode.splitToString(split, left = false)})\n" +
      rightChild.subtreeToString(indentFactor + 1)
  }
}

object InternalNode {

  /**
   * Helper method for [[Node.subtreeToString()]].
   *
   * @param split Split to print
   * @param left  Indicates whether this is the part of the split going to the left,
   *              or that going to the right.
   */
  private def splitToString(split: Split, left: Boolean): String = {
    val featureStr = s"feature ${split.featureIndex}"
    split match {
      case contSplit: ContinuousSplit =>
        if (left) {
          s"$featureStr <= ${contSplit.threshold}"
        } else {
          s"$featureStr > ${contSplit.threshold}"
        }
      case catSplit: CategoricalSplit =>
        val categoriesStr = catSplit.leftCategories.mkString("{", ",", "}")
        if (left) {
          s"$featureStr in $categoriesStr"
        } else {
          s"$featureStr not in $categoriesStr"
        }
    }
  }
}