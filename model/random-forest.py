from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.util import MLUtils

sc = SparkContext("local", "Data")
fName = "file:////Users/baobao/Desktop/data_format/normalize_data.csv"

def extract(line):
	if line == raw_data_header:
		return False
	for x in line.split(','):
		if x == '""' or x=='-1':
			return False
	return True

def parsePoint(line):
	values = [float(x) for x in line.split(',')]
	return LabeledPoint(values[-1], values[:-1])

raw_data = sc.textFile(fName)
raw_data_header = raw_data.take(1)[0]
parsedData = raw_data.filter(extract).map(parsePoint)

(trainingData, testData) = parsedData.randomSplit([0.7, 0.3])

# Train a RandomForest model.
model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=3, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
# testErr = labelsAndPredictions.filter(
#     lambda lp: lp[0] != lp[1]).count() / float(testData.count())
# print('Test Error = ' + str(testErr))
# print('Learned classification forest model:')
# print(model.toDebugString())

RepeatedCus = labelsAndPredictions.filter(lambda lp: lp[0] == 1.0)
trainErr = RepeatedCus.filter(lambda lp: lp[0] != lp[1]).count()/float(RepeatedCus.count())
print("Training Error = " + str(trainErr))

metrics = BinaryClassificationMetrics(labelsAndPredictions)
# Area under ROC curve
print("Area under ROC = %s" % metrics.areaUnderROC)
# Area under precision-recall curve
print("Area under PR = %s" % metrics.areaUnderPR)


