package bart.tree

import bart.discretizer.{CategoricalSplit, ContinuousSplit, Split, TreePoint}
import bart.ParArrOrSeqArr

/**
 * Version of a node used in learning.  This uses vars so that we can modify nodes as we split the
 * tree by adding children, etc.
 *
 * For now, we use node IDs.  These will be kept internal since we hope to remove node IDs
 * in the future, or at least change the indexing (so that we can support much deeper trees).
 *
 * This node can either be:
 *  - a leaf node, with leftChild, rightChild, split set to null, or
 *  - an internal node, with all values set
 *
 * @param id  We currently use the same indexing as the old implementation in
 *            [[org.apache.spark.mllib.tree.model.Node]], but this will change later.
 */
class LearningNode(
  val id: Int,
  var leftChild: Option[LearningNode] = None,
  var rightChild: Option[LearningNode] = None,
  var split: Option[Split] = None,
  val sufficientStat: SufficientStat,
  var df: Option[ParArrOrSeqArr[TreePoint]] = None) extends Serializable {

  override def toString: String = s"nodeId: $id"

  def isLeaf: Boolean = leftChild.isEmpty && rightChild.isEmpty

  /**
  singly internal: both children nodes are terminal nodes, used in change and prune proposal
   */
  def isSinglyInternal: Boolean = {
    (!isLeaf) && leftChild.get.isLeaf && rightChild.get.isLeaf && split.isDefined
  }

  def allLeafNodes: Array[LearningNode] = {
    if (isLeaf) {
      Array(this)
    } else {
      leftChild.get.allLeafNodes ++ rightChild.get.allLeafNodes
    }
  }

  def allSinglyInternalNodes: Array[LearningNode] = {
    if (isLeaf) {
      Array[LearningNode]()
    } else if (isSinglyInternal) {
      Array(this)
    } else {
      leftChild.get.allSinglyInternalNodes ++ rightChild.get.allSinglyInternalNodes
    }
  }

  /**
   * Convert this [[LearningNode]] to a regular [[Node]], and recurse on any children.
   */
  def toNode(splits: Array[Array[Split]]): Node = {

    assert(leftChild.nonEmpty == rightChild.nonEmpty &&
      leftChild.nonEmpty == split.nonEmpty,
      "leftChild, rightChild, split and sufficientStat must all be none or not none")

    if (leftChild.nonEmpty) {

      assert(sufficientStat != null,
        "Unknown error during Decision Tree learning. Could not convert LearningNode to Node.")
      (leftChild.get.toNode(splits), rightChild.get.toNode(splits)) match {
        case (l, r) =>
          val newSplit = split.get match {
            case con: ContinuousSplit =>
              splits(con.featureIndex)(con.binnedThreshold)
            case cat: CategoricalSplit =>
              cat
          }
          new InternalNode(l, r, newSplit)
      }

    } else {
      new LeafNode(sufficientStat.leafResp)
    }
  }

  /**
   * Get the leaf resp corresponding to this data point.
   * This function mimics prediction, passing an example from the root node down to a leaf
   * or unsplit node; that node's index is returned.
   *
   * @param binnedFeatures  Binned feature vector for data point.
   * @return Leaf resp
   */
  def predict(binnedFeatures: Array[Int]): Double = {
    var node = this
    while (!node.isLeaf) {
      val split = node.split.get
      if (split.shouldGoLeft(binnedFeatures)) {
        node = node.leftChild.get
      } else {
        node = node.rightChild.get
      }
    }
    node.sufficientStat.leafResp
  }

  val nodeIdPath: Array[Int] = {
    val maxLevel = LearningNode.indexToLevel(id)
    var iLevel =  maxLevel - 1
    var iNodeId = id
    val nodeIds = Array.fill(maxLevel) {id}
    while (iLevel > 0) {
      iNodeId = LearningNode.parentIndex(iNodeId)
      iLevel = iLevel - 1
      nodeIds(iLevel) = iNodeId
    }
    nodeIds
  }

  def copy(): LearningNode = {
    LearningNode(id, sufficientStat.copy())
  }

  def deepCopy(newChainId: Int): LearningNode = {
    if (leftChild.nonEmpty) {
      assert(leftChild.nonEmpty && rightChild.nonEmpty && split.nonEmpty,
        "Unknown error during Decision Tree learning.  Could not convert LearningNode to Node.")
      val (l, r) = (
        leftChild.get.deepCopy(newChainId),
        rightChild.get.deepCopy(newChainId))
      new LearningNode(this.id, Some(l), Some(r), this.split, this.sufficientStat.copy(newChainId), None)
    } else {
      new LearningNode(this.id, None, None, None, this.sufficientStat.copy(newChainId), None)
    }

  }
}

object LearningNode {

  /** Create a node with some of its fields set. */
  def apply(
    id: Int,
    sufficientStat: SufficientStat): LearningNode = {
    new LearningNode(id, None, None, None, sufficientStat, None)
  }

  /**
   * Return the index of the left child of this node.
   */
  def leftChildIndex(nodeIndex: Int): Int = nodeIndex << 1

  /**
   * Return the index of the right child of this node.
   */
  def rightChildIndex(nodeIndex: Int): Int = (nodeIndex << 1) + 1

  /**
   * Get the parent index of the given node, or 0 if it is the root.
   */
  def parentIndex(nodeIndex: Int): Int = nodeIndex >> 1

  /**
   * Return the level of a tree which the given node is in.
   */
  def indexToLevel(nodeIndex: Int): Int = if (nodeIndex == 0) {
    throw new IllegalArgumentException(s"0 is not a valid node index.")
  } else {
    java.lang.Integer.numberOfTrailingZeros(java.lang.Integer.highestOneBit(nodeIndex))
  }

  /**
   * Returns true if this is a left child.
   * Note: Returns false for the root.
   */
  def isLeftChild(nodeIndex: Int): Boolean = nodeIndex > 1 && nodeIndex % 2 == 0

  /**
   * Traces down from a root node to get the node with the given node index.
   * This assumes the node exists.
   */
  def getNode(nodeIndex: Int, rootNode: LearningNode): LearningNode = {
    var tmpNode: LearningNode = rootNode
    var levelsToGo = indexToLevel(nodeIndex)
    while (levelsToGo > 0) {
      if ((nodeIndex & (1 << levelsToGo - 1)) == 0) {
        tmpNode = tmpNode.leftChild.get
      } else {
        tmpNode = tmpNode.rightChild.get
      }
      levelsToGo -= 1
    }
    tmpNode
  }

  def isBelongTargetNode(
    rootNode: LearningNode,
    targetNode: LearningNode,
    binnedFeatures: Array[Int]): Boolean = {
    val maxLevel = targetNode.nodeIdPath.length
    var node = rootNode
    var i = 0
    while (i < maxLevel) {
      val shouldLeft = targetNode.nodeIdPath(i) % 2 == 0
      val split = node.split.get
      val featureIndex = split.featureIndex
      val splitLeft = split.shouldGoLeft(binnedFeatures(featureIndex))
      if (shouldLeft != splitLeft) {
        return false
      } else if (shouldLeft) {
        node = node.leftChild.get
      } else {
        node = node.rightChild.get
      }
      i = i + 1
    }
    true
  }
}