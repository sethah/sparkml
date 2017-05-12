package com.sethah.spark

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.mllib.random.{RandomRDDs, RandomDataGenerator}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

class MultinomialDataGenerator(
                                private val coefficients: Array[Double],
                                private val xMean: Array[Double],
                                private val xVariance: Array[Double],
                                private val addIntercept: Boolean,
                                private val seed: Long) extends RandomDataGenerator[LabeledPoint] {

  val xDim = xMean.length
  val xWithInterceptsDim = if (addIntercept) xDim + 1 else xDim
  val nClasses = coefficients.length / xWithInterceptsDim + 1

  private val rng = new java.util.Random(seed)

  override def nextValue(): LabeledPoint = {
    val x = Array.tabulate(xDim) { i =>
      rng.nextGaussian() * math.sqrt(xVariance(i)) + xMean(i)
    }
    val y = {
      val margins = Array.ofDim[Double](nClasses)
      val probs = Array.ofDim[Double](nClasses)

      for (i <- 0 until nClasses - 1) {
        for (j <- 0 until xDim) margins(i + 1) += coefficients(i * xWithInterceptsDim + j) * x(j)
        if (addIntercept) margins(i + 1) += coefficients((i + 1) * xWithInterceptsDim - 1)
      }
      // Preventing the overflow when we compute the probability
      val maxMargin = margins.max
      if (maxMargin > 0) for (i <- 0 until nClasses) margins(i) -= maxMargin


      // Computing the probabilities for each class from the margins.
      val norm = {
        var temp = 0.0
        for (i <- 0 until nClasses) {
          probs(i) = math.exp(margins(i))
          temp += probs(i)
        }
        temp
      }
      for (i <- 0 until nClasses) probs(i) /= norm

      // Compute the cumulative probability so we can generate a random number and assign a label.
      for (i <- 1 until nClasses) probs(i) += probs(i - 1)
      val p = rng.nextDouble()
      var prob = probs(0)
      var i = 0
      while (p > prob) {
        i += 1
        prob = probs(i)
      }
      i
    }
    LabeledPoint(y.toDouble, Vectors.dense(x))
  }

  def setSeed(seed: Long) {
    rng.setSeed(seed)
  }

  override def copy(): MultinomialDataGenerator =
    new MultinomialDataGenerator(coefficients, xMean, xVariance, addIntercept, seed)

}

object MultinomialDataGenerator {
  def makeData(
                spark: SparkSession,
                numClasses: Int,
                numFeatures: Int,
                fitIntercept: Boolean,
                numPoints: Int,
                seed: Long): RDD[LabeledPoint] = {
    val rng = scala.util.Random
    rng.setSeed(seed)
    val coefWithInterceptLength = if (fitIntercept) numFeatures + 1 else numFeatures
    val coefficients = Array.tabulate((numClasses - 1) * coefWithInterceptLength) { i =>
      rng.nextDouble() - 0.5
    }
    val xMean = Array.tabulate(numFeatures) { i => (rng.nextDouble() - 0.5) * 5}
    val xVariance = Array.tabulate(numFeatures) { i => rng.nextDouble() * 2 + 1}
    val generator = new MultinomialDataGenerator(coefficients, xMean, xVariance, fitIntercept, seed)
    RandomRDDs.randomRDD(spark.sparkContext,
      generator, numPoints, spark.sparkContext.defaultParallelism, seed)
  }
}
