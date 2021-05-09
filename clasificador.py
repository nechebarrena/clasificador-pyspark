import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import pyspark
from pyspark.sql import SparkSession
print("-- CLASIFICADOR --")


iris_data = datasets.load_iris()
df_iris = pd.DataFrame(iris_data.data,columns=iris_data.feature_names)
df_iris['target'] = pd.Series(iris_data.target)
print(df_iris.head())
etiquetas = df_iris['target'].unique()
print(etiquetas)


#Create PySpark SparkSession
spark = SparkSession.builder \
    .master("local[1]") \
    .appName("CLASIFICADOR") \
    .getOrCreate()
#Create PySpark DataFrame from Pandas
sparkDF=spark.createDataFrame(df_iris)
sparkDF.printSchema()
sparkDF.show()


from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

# transformer
print("NUEVO")
vector_assembler = VectorAssembler(inputCols=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],outputCol="features")
sparkDF = vector_assembler.transform(sparkDF)
sparkDF.show(10)

(trainingData, testData) = sparkDF.randomSplit([0.7, 0.3])
print("Training >>>")
trainingData.show(2)

print("Test >>>")
testData.show(2)

rf = RandomForestClassifier(labelCol='target', featuresCol="features", numTrees=10)
model = rf.fit(trainingData)

# test our model and make predictions using testing data
predictions = model.transform(testData)
predictions.select("prediction","target", "features").show(5)

evaluator = MulticlassClassificationEvaluator(labelCol="features", predictionCol="prediction",metricName="accuracy")


y_true = predictions.select(['target']).collect()
y_pred = predictions.select(['prediction']).collect()


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)

print(cm)

clases = ['0','1','2']
from sklearn.metrics import precision_score, recall_score, classification_report
reporte = classification_report(y_true, y_pred, target_names=clases)
print(reporte)
