/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.apache.spark.ml.util.{DefaultParamsReader, DefaultParamsWriter}
import org.apache.spark.sql.SparkSession
import org.json4s.{DefaultFormats, JObject}

/**
 * Abstraction for Bayes Additive Tree models.
 */
private[spark] trait BayesAdditiveTreeModel {

  /** Root of the decision tree */
  val multiForest: Seq[Seq[bart.tree.Node]]


  /** Summary of the model */
  override def toString: String = {
    // Implementing classes should generally override this method to be more descriptive.
    //s"BayesAdditiveTreeModel of depth $depth with $numNodes nodes"
    s"BayesAdditiveTreeModel of depth"
  }

//  /** Full description of model */
//  def toDebugString: String = {
//    val header = toString + "\n"
//    header + rootNode.subtreeToString(2)
//  }

}


/** Helper classes for tree model persistence */
private[ml] object BayesAdditiveTreeModelReadWrite {
  /**
   * Load a decision tree from a file.
   * @return  Root node of reconstructed tree
   */
  def loadForests(
      path: String,
      sparkSession: SparkSession): Seq[Seq[bart.tree.Node]] = {

    import sparkSession.implicits._
    implicit val format: DefaultFormats.type = DefaultFormats
    val dataPath = new Path(path, "data").toString
    val df = sparkSession.read.parquet(dataPath)
    df.as[ForestData].map(f => ForestData.getForest(f)).collect().toSeq
  }


  /**
   * Helper method for saving a tree ensemble to disk.
   *
   * @param instance  Tree ensemble model
   * @param path  Path to which to save the ensemble model.
   * @param extraMetadata  Metadata such as numFeatures, numClasses, numTrees.
   */
  def saveImpl[M <: Params with BayesAdditiveTreeModel](
    instance: M,
    path: String,
    sql: SparkSession,
    extraMetadata: JObject): Unit = {

    DefaultParamsWriter.saveMetadata(instance, path, sql.sparkContext, Some(extraMetadata))
    val dataPath = new Path(path, "data").toString
    val forests = instance.multiForest.map(f => ForestData(f.map(t => NodeData.build(t, 0)._1)))
    sql.createDataFrame(forests).toDF("forest")
      .write.parquet(dataPath)
  }

  /**
   * Helper method for loading a tree ensemble from disk.
   * This reconstructs all trees, returning the root nodes.
   * @param path  Path given to `saveImpl`
   * @param className  Class name for ensemble model type
   * @return  (ensemble metadata, array over trees of (tree metadata, root node)),
   *          where the root node is linked with all descendents
   * @see `saveImpl` for how the model was saved
   */
  def loadImpl(
    path: String,
    sql: SparkSession,
    className: String
  ): (Metadata, Seq[Seq[bart.tree.Node]]) = {
    import sql.implicits._
    implicit val format: DefaultFormats.type = DefaultFormats
    val metadata = DefaultParamsReader.loadMetadata(path, sql.sparkContext, className)

    val dataPath = new Path(path, "data").toString
    val df = sql.read.parquet(dataPath).as[ForestData].
      map(f => ForestData.getForest(f)).collect().toSeq
    (metadata, df)
  }
}

