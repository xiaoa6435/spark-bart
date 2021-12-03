package org.apache.spark.ml

case class ForestData(forest: Seq[Seq[NodeData]])

object ForestData {
  def apply(forest: Seq[bart.tree.Node]): Seq[Seq[NodeData]] = {
    forest.map(t => NodeData.build(t, 0)._1)
  }

  def getForest(forestData: ForestData): Seq[bart.tree.Node] = {
    forestData.forest.map(f => NodeData.getNode(f))
  }
}
