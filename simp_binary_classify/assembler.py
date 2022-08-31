from pyspark.ml.feature import OneHotEncoder, VectorAssembler
import pandas as pd

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def assemble_feature(train_data):
    def __init__(self):
        # get transformed train and test data
        yvals_DF = spark.createDataFrame(
            pd.DataFrame(
                {
                    "row_num": [i for i in range(train_data.shape[0])],
                    "yvals": train_data["class_col"]
                }
            )
        )

        # (
        #     sc.parallelize([Row(x) for x in range(train_data)])
        #     .toDF()
        #     .withColumnRenamed("_1", "row_num")
        # )
        yvals_DF.createOrReplaceTempView("yvals")


# get columns with _cat in name
cat_cols = list(filter(lambda nm: nm.find("_cat") != -1, trainDF.columns))
updated_cols = list(filter(lambda nm: nm.find("_cat") == -1, trainDF.columns))
for c in cat_cols:
    encoder = OneHotEncoder(inputCol=c, outputCol = c + '_vec')
    onehotdata = encoder.fit(trainDF).transform(trainDF)
    onehotdata.show(10, truncate=False)


assembler = VectorAssembler(
    inputCols=trainDF.columns[1:],
    outputCol="features"
)

outdata = assembler.transform(trainDF)
outdata.createOrReplaceTempView("out")

testDF = spark.sql("SELECT * FROM test")
testoutdata = assembler.transform(testDF)
testoutdata.createOrReplaceTempView("testout")
testDF = spark.sql("SELECT row_num as id, features from testout")

lr = LogisticRegression(maxIter=10)
pipeline = Pipeline(stages=[lr])

paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .build()

modelDataDF = spark.sql("SELECT a.row_num as id, a.yvals as label, b.features FROM yvals a LEFT JOIN out b ON a.row_num = b.row_num")

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=2)

cvModel = crossval.fit(modelDataDF)

# Make predictions on test documents. cvModel uses the best model found (lrModel).
prediction = cvModel.transform(testDF)
selected = prediction.select("id", "features", "probability", "prediction")
for row in selected.collect():
    print(row)
