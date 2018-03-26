import org.apache.spark.mllib.evaluation.MulticlassMetrics
import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.feature.{QuantileDiscretizer, VectorAssembler, VectorIndexer}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
  object xgboost {
    def main(args: Array[String]): Unit = {
      Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
      Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
      val spark = SparkSession.builder.master("local").appName("example").

        config("spark.sql.shuffle.partitions", "20").getOrCreate()
      spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      val path = "file:///home/supper/balanced_train.csv"

      val Data = spark.read.option("header", true).option("inferSchema",true).csv(path)

      val DroppedData = Data.na.drop().filter(Data("label") >= 0)

//      DroppedData.show(10)
//      println(DroppedData.columns.length)
//      val indexer = new VectorIndexer()
//        .setInputCol("features")
//        .setOutputCol("indexed")
//        .setMaxCategories(10)

//      val discretizers = new Array[QuantileDiscretizer](DroppedData.columns.length - 6)
//      val outPutColumns = Array("DR", "DMACM", "DMACB", "DMACC", "DMACI", "DUMA", "DUAD", "DUIM", "DUCM"
//                                , "DUBM", "DUME", "DUUI", "DUUC", "DUUB", "DUUM", "DAD", "DUPIT", "DUPBT", "DUPCT", "DUPMT")
//      val stages = Array
//      for (i <- 5 until DroppedData.columns.length - 1){
//        discretizers(i - 5) = new QuantileDiscretizer().setInputCol(DroppedData.columns(i)).setOutputCol(outPutColumns(i - 5))
//      }
//      val assembler = new VectorAssembler()
//        .setInputCols(Array("age_range", "gender","DR", "DMACM", "DMACB", "DMACC", "DMACI", "DUMA", "DUAD", "DUIM", "DUCM"
//          , "DUBM", "DUME", "DUUI", "DUUC", "DUUB", "DUUM", "DAD", "DUPIT", "DUPBT", "DUPCT", "DUPMT"))
//        .setOutputCol("features")
      val assemblerNonDisc = new VectorAssembler().setInputCols(DroppedData.columns.filter(col => !col.contains("label")).
                                filter(col => !col.contains("id"))).setOutputCol("featuresOri")
      val cleaned = assemblerNonDisc.transform(DroppedData)
      cleaned.show(10)
//      val pipeline = new Pipeline().setStages(discretizers)
//      val processed = pipeline.fit(DroppedData).transform(DroppedData)
//      processed.show(10)
//      val pipeline2 = new Pipeline().setStages(Array(assembler, indexer))
//      val cleaned = pipeline2.fit(processed).transform(processed)
      val Array(train, test) = cleaned.randomSplit(Array(0.8, 0.2))
//      //////////////////////////
//      val Array(trainBug, testBug) = test.randomSplit(Array(0.8, 0.2))
//      val Array(bugtrain, bugtest) = debugData.randomSplit(Array(0.8, 0.2))
      /////////////////////////

      val numRound = 3
      val paramMap = List(
        "eta" -> 0.1,
        "min_child_weight" -> 5,
        "max_depth" -> 5,
        "silent" -> 1,
        "objective" -> "binary:logistic",
        "lambda" -> 1,
        "nthread" -> 1
      ).toMap

      val model = XGBoost.trainWithDataFrame(train, paramMap, 1, 1, obj = null, eval = null, useExternalMemory = false, Float.NaN, "featuresOri", "label")
      val test1 = test.withColumn("label", test.col("label").cast(DoubleType))
      val predict = model.transform(test1)

      val scoreAndLabels = predict.select(model.getPredictionCol, model.getLabelCol)
        .rdd
        .map{case Row(score:Double, label:Double) => (score, label)}
      //get the auc

      val metric = new MulticlassMetrics(scoreAndLabels)

      println(metric.confusionMatrix)

    }
  }
