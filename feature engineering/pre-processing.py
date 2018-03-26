from pyspark import SparkContext
from pyspark import SparkConf
from pyspark import sql
from pyspark import RDD
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

# define function
def trans_action(l):
	if l[7] == '1':
		l[7] = '2'
	elif l[7] == '2':
		l[7] = '3'
	elif l[7] == '3':
		l[7] = '1'
	return l

def toCSVLine(data):
  return ','.join(str(d) for d in data)

# parse the data
sc = SparkContext("local", "Data")
fName = "file:////Users/baobao/Desktop/data_format/user_log.csv"

raw_data = sc.textFile(fName)
raw_data_header = raw_data.take(1)[0]
parsedData = raw_data.filter(lambda line: line!=raw_data_header).map(lambda line:line.split(','))

newDate = parsedData.map(trans_action).map(lambda l: ((int(l[0]), int(l[3])), (float(l[7]), 1.0)))
averageData = newDate.reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1])).map(lambda l: (l[0][0], l[0][1], l[1][0]/l[1][1]))

ratings = averageData.map(lambda l: Rating(l[0], l[1], l[2]))

# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)

# Evaluate the model on training data
testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
result = ratesAndPreds.map(lambda l: (l[0][0], l[0][1], l[1][1]))
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print(result.take(10))

lines = result.map(toCSVLine)
lines.saveAsTextFile('output')
# print("Mean Squared Error = " + str(MSE))

# /usr/local/spark/bin/spark-submit --master local[4] data.py