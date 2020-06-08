from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import IntegerType
from pyspark.sql.types import FloatType



sc = SparkContext('local')
spark = SparkSession(sc)
# Load training data
data = spark.read.format("csv").option("header", "true") \
    .load("data/guns_type_training.csv")

data = data.withColumn("n_males", data["n_males"].cast(IntegerType()))
data = data.withColumn("n_females", data["n_females"].cast(IntegerType()))
data = data.withColumn("n_victims", data["n_victims"].cast(IntegerType()))
data = data.withColumn("n_suspects", data["n_suspects"].cast(IntegerType()))
data = data.withColumn("n_unharmed", data["n_unharmed"].cast(IntegerType()))
data = data.withColumn("n_arrested", data["n_arrested"].cast(IntegerType()))
data = data.withColumn("n_killed", data["n_killed"].cast(IntegerType()))
data = data.withColumn("n_injured", data["n_injured"].cast(IntegerType()))
data = data.withColumn("latitude", data["latitude"].cast(FloatType()))
data = data.withColumn("longitude", data["longitude"].cast(FloatType()))
data = data.withColumn("n_guns_involved", data["n_guns_involved"].cast(IntegerType()))
data = data.withColumn("label", data["gt_Unknown"].cast(FloatType()))


feature_set = ["n_killed", "n_injured","n_guns_involved", "n_males", "n_females", "n_victims", "n_suspects", "n_unharmed", "n_arrested" ]

vecAssembler = VectorAssembler(inputCols=feature_set, outputCol="features")
df = vecAssembler.setHandleInvalid("skip").transform(data)

# Split the data into train and test
splits = df.randomSplit([0.8, 0.2], 1234)
train = splits[0]
test = splits[1]



data = spark.read.format("csv").option("header", "true") \
    .load("data/guns_type_classification.csv")
data = data.withColumn("n_males", data["n_males"].cast(IntegerType()))
data = data.withColumn("n_females", data["n_females"].cast(IntegerType()))
data = data.withColumn("n_victims", data["n_victims"].cast(IntegerType()))
data = data.withColumn("n_suspects", data["n_suspects"].cast(IntegerType()))
data = data.withColumn("n_unharmed", data["n_unharmed"].cast(IntegerType()))
data = data.withColumn("n_arrested", data["n_arrested"].cast(IntegerType()))
data = data.withColumn("n_killed", data["n_killed"].cast(IntegerType()))
data = data.withColumn("n_injured", data["n_injured"].cast(IntegerType()))
data = data.withColumn("latitude", data["latitude"].cast(FloatType()))
data = data.withColumn("longitude", data["longitude"].cast(FloatType()))
data = data.withColumn("n_guns_involved", data["n_guns_involved"].cast(IntegerType()))
data = data.withColumn("label", data["gun_type"].cast(FloatType()))
predict = data
print(predict)

# create the trainer and set its parameters
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# train the model
model = nb.fit(train)

# select example rows to display.
predictions = model.transform(test)
predictions.show()
# compute accuracy on the test set
guntypes = ["gt_Unknown",  "n_guns"    ,"n_guns_not_stolen"  ,"n_guns_stolen"    ,"gt_22 LR"    ,"gt_223 Rem [AR-15]" ,"gt_Shotgun","gt_9mm","gt_45 Auto"   ,"gt_12 gauge"   ,"gt_7.62 [AK-47]"  ,"gt_40 SW","gt_44 Mag"     ,                 "gt_Other"      ,                 "gt_38 Spl"       ,               "gt_380 Auto"    ,                "gt_32 Auto"       ,              "gt_410 gauge"     ,              "gt_308 Win"       ,              "gt_Rifle"          ,             "gt_357 Mag"       ,              "gt_16 gauge"      ,              "gt_30-30 Win"     ,              "gt_25 Auto"       ,              "gt_20 gauge"      ,              "gt_10mm"          ,              "gt_30-06 Spr"     ,              "gt_300 Win"       ,              "gt_28 gauge"  ]                  

for gun in guntypes :
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Training set accuracy for"+gun+" = " + str(accuracy))

vecAssembler = VectorAssembler(inputCols=feature_set, outputCol="features")
df = vecAssembler.setHandleInvalid("skip").transform(predict)
predictions = model.transform(df)
predictions.show()
predictions.toPandas().to_csv('predictions_gun_type.csv')
print("Predictions written to file predictions_gun_type.csv")
