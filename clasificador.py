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

(trainingData, testData) = sparkDF.randomSplit([0.7, 0.3])
print("Training >>>")
trainingData.show()

print("Test >>>")
testData.show()

features_col = ['petal length (cm)','petal width (cm)']
target_col = ['target']

rf = RandomForestClassifier(labelCol='target', featuresCol=features_col, numTrees=10)
rfModel = rf.fit(trainingData)


trainingSummary = rfModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))

