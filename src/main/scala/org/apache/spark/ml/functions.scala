package org.apache.spark.ml

import org.apache.spark.sql.functions.udf

object functions {
  private[ml] def checkNonNegativeWeight = udf {
    value: Double =>
      require(value >= 0, s"illegal weight value: $value. weight must be >= 0.0.")
      value
  }
}
