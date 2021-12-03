package org.apache.spark.ml

import bart.tree.InternalNode

/**
 * Info for a [[bart.tree.Node]]
 *
 * @param id  Index used for tree reconstruction.  Indices follow a pre-order traversal.
 * @param leftChild  Left child index, or arbitrary value if leaf node.
 * @param rightChild  Right child index, or arbitrary value if leaf node.
 * @param split  Split info, or arbitrary value if leaf node.
 */
case class NodeData(
  id: Int,
  prediction: Double,
  leftChild: Int,
  rightChild: Int,
  split: SplitData)

object NodeData {
  /**
   * Create [[NodeData]] instances for this node and all children.
   *
   * @param id  Current ID.  IDs are assigned via a pre-order traversal.
   * @return (sequence of nodes in pre-order traversal order, largest ID in subtree)
   *         The nodes are returned in pre-order traversal (root first) so that it is easy to
   *         get the ID of the subtree's root node.
   */
  def build(node: bart.tree.Node, id: Int): (Seq[NodeData], Int) = node match {
    case n: bart.tree.InternalNode =>
      val (leftNodeData, leftIdx) = build(n.leftChild, id + 1)
      val (rightNodeData, rightIdx) = build(n.rightChild, leftIdx + 1)
      val thisNodeData = NodeData(id, -1.0, leftNodeData.head.id, rightNodeData.head.id,
        SplitData(n.split))
      (thisNodeData +: (leftNodeData ++ rightNodeData), rightIdx)
    case n: bart.tree.LeafNode =>
      (Seq(NodeData(id, n.mu, -1, -1, new SplitData(-1, Array.emptyDoubleArray, -1))),
        id)
  }

  def getNode(data: Seq[NodeData]): bart.tree.Node = {
    // Load all nodes, sorted by ID.
    val nodes = data.sortBy(_.id)
    // Sanity checks; could remove
    assert(nodes.head.id == 0, s"Decision Tree load failed.  Expected smallest node ID to be 0," +
      s" but found ${nodes.head.id}")
    assert(nodes.last.id == nodes.length - 1, s"Decision Tree load failed.  Expected largest" +
      s" node ID to be ${nodes.length - 1}, but found ${nodes.last.id}")
    // We fill `finalNodes` in reverse order.  Since node IDs are assigned via a pre-order
    // traversal, this guarantees that child nodes will be built before parent nodes.
    val finalNodes = new Array[bart.tree.Node](nodes.length)
    nodes.reverseIterator.foreach { n: NodeData =>

      val node = if (n.leftChild != -1) {
        val leftChild = finalNodes(n.leftChild)
        val rightChild = finalNodes(n.rightChild)
        new InternalNode(leftChild, rightChild, n.split.getSplit)
      } else {
        new bart.tree.LeafNode(n.prediction)
      }
      finalNodes(n.id) = node
    }
    // Return the root node
    finalNodes.head
  }
}

