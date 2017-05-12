package com.sethah.spark

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.{LogReg, LogisticAggregator, LogisticRegression}
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{Matrices, Vectors}
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.{Exp, Log}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.util.FeatureUtil
import scopt.OptionParser

object Main {
  private[this] case class Params(
    numClasses: Int = 10,
    numFeatures: Int = 10,
    numExamples: Int = 1000,
    regParam: Double = 0.0,
    maxIter: Int = 100,
    test: String = "",
    seed: Long = 42L,
    fitIntercept: Boolean = true)


  private[this] object Params {
    def parseArgs(args: Array[String]): Params = {
      val params = new OptionParser[Params]("Load and save jpegs") {
        opt[Int]("numClasses")
          .text("asdf")
          .action((x, c) => c.copy(numClasses = x))
        opt[Int]("numFeatures")
          .action((x, c) => c.copy(numFeatures = x))
        opt[Int]("numExamples")
          .action((x, c) => c.copy(numExamples = x))
        opt[Double]("regParam")
          .action((x, c) => c.copy(regParam = x))
        opt[Int]("maxIter")
          .action((x, c) => c.copy(maxIter = x))
        opt[Long]("seed")
          .action((x, c) => c.copy(seed = x))
        opt[String]("test")
          .action((x, c) => c.copy(test = x))
        opt[Boolean]("fitIntercept")
          .action((x, c) => c.copy(fitIntercept = x))
      }.parse(args, Params()).get
      params
    }
  }

  def main(args: Array[String]): Unit = {
    val params = Params.parseArgs(args)
    val sparkConf = new SparkConf().setAppName("test logistic regression").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)
    org.apache.log4j.Logger.getRootLogger().setLevel(org.apache.log4j.Level.ERROR)
    try {
      val spark = SparkSession.builder().getOrCreate()
      import spark.sqlContext.implicits._
      val rdd = MultinomialDataGenerator.makeData(spark, params.numClasses, params.numFeatures,
        params.fitIntercept, params.numExamples, params.seed)
      rdd.take(12).foreach(println)
      val df = rdd.toDF()

      params.test match {
        case "spark" =>
          val lr = new LogisticRegression()
            .setMaxIter(params.maxIter)
            .setRegParam(params.regParam)
          val t0 = System.nanoTime()
          val model = lr.fit(df)
          val t1 = System.nanoTime()
          println(model.coefficientMatrix)
          println((t1 - t0) / 1e9)
        case "nd4" =>
          val lr = new LogReg()
            .setMaxIter(params.maxIter)
            .setRegParam(params.regParam)
          val t0 = System.nanoTime()
          val model = lr.fit(df)
          val t1 = System.nanoTime()
          println(model.coefficientMatrix)
          println((t1 - t0) / 1e9)
        case other =>
          println(s"Test $other not supported. Ignoring.")
      }

    } finally {
      sc.stop()
    }
  }
}
