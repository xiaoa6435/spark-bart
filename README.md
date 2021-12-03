# spark-bart
A pure scala/spark implementation of the BART(bayes additive regressions trees model of [Chipman et al](https://arxiv.org/abs/0806.3286) and related model, like BACT(bayes additive classification tree), BCF(bayes causal forest) etc

this project is currently **working in progress**

# example
```
import org.apache.spark.ml.linalg.Vectors
import math.{Pi, sin}
import util.Random

import org.apache.spark.ml.regression.{BayesAdditiveTreeRegressionModel, BayesAdditiveTreeRegressor}
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator

val sampleCnt = (1e3).toInt
val p = 10

// Friedman’s test function with a category 
val df = spark.createDataFrame(
  Range(0, sampleCnt).map{i => 
    val x = Array.tabulate(p){i => 
      if(i == 9) Random.nextInt(5).toDouble else Random.nextDouble
    }
    val features = Vectors.dense(x)
    val label = (10 * math.sin(math.Pi * x(0) * x(1)) + 
             20 * math.pow(x(2) - 0.5, 2) + 
             10 * x(3) + 5 * x(4) + Random.nextGaussian)
  (features, label)
}).toDF("features", "label")

val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))
val BART = new BayesAdditiveTreeRegressor().
  setNumBurn(1000).
  setNumSim(100).
  setParallelChainCnt(4).
  setNumThin(2).
  setCategoryFeatureIndexes(Array(9)).
  setCategoryFeatureArity(Array(5))
val bartModel = BART.fit(trainingData)
val bartPred = bartModel.transform(testData)

val evaluator = new RegressionEvaluator().
  setLabelCol("label").
  setPredictionCol("prediction").
  setMetricName("rmse")

val bartRMSE = evaluator.evaluate(bartPred)
println(s"bart: Root Mean Squared Error (RMSE) on test data = $bartRMSE")
//bart: Root Mean Squared Error (RMSE) on test data = 1.83

val GBT = new GBTRegressor()
val GBTModel = GBT.fit(trainingData)
val gbtPred = GBTModel.transform(testData)
val gbtRMSE = evaluator.evaluate(gbtPred)
println(s"gbt: Root Mean Squared Error (RMSE) on test data = $gbtRMSE")
//gbt: Root Mean Squared Error (RMSE) on test data = 2.31

bartModel.asInstanceOf[BayesAdditiveTreeRegressionModel].
  write.overwrite.save("bart-model")
```

# Related projects

- [bartMachine](https://github.com/kapelner/bartMachine)
- [bartpy](https://github.com/JakeColtman/bartpy)
- [bcf](https://github.com/jaredsmurray/bcf)

# References
- [BART: Bayesian additive regression trees](https://arxiv.org/abs/0806.3286)
- [Bayesian regression tree models for causal inference](https://arxiv.org/abs/1706.09523)
- [Bayes and Big Data: The Consensus Monte Carlo Algorithm](https://research.google/pubs/pub41849/)
