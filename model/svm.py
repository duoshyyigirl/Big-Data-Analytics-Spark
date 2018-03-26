from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.classification import SVMWithSGD,SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel

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

# Split data aproximately into training (70%) and test (30%)
training, test = parsedData.randomSplit([0.7, 0.3], seed=0)

# Build the model
model = SVMWithSGD.train(training, iterations=1000)
#model = LogisticRegressionWithLBFGS.train(training)

# Evaluating the model on training data
PredsAndlabels = test.map(lambda p: (float(model.predict(p.features)), p.label))

TotaltrainErr = PredsAndlabels.filter(lambda lp: lp[0] != lp[1]).count() / float(test.count())
print("Test Error for all customers = " + str(TotaltrainErr))

RepeatedCus = PredsAndlabels.filter(lambda lp: lp[1] == 1.0)
trainErr = RepeatedCus.filter(lambda lp: lp[0] != lp[1]).count()/float(RepeatedCus.count())
print("Test Error for repeated customers = " + str(trainErr))

metrics = BinaryClassificationMetrics(PredsAndlabels)
# Area under ROC curve
print("Area under ROC = %s" % metrics.areaUnderROC)
# Area under precision-recall curve
print("Area under PR = %s" % metrics.areaUnderPR)

