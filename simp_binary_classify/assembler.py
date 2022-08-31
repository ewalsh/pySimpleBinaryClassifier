from pyspark.ml.feature import OneHotEncoder, VectorAssembler
import pandas as pd
from simp_binary_classify.spark import spark


def assemble_features(train_data):
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
        # add to spark sql
        yvals_DF.createOrReplaceTempView("yvals")

    # create assembler function
    def gen_assembly():
        trainDF = spark.sql("SELECT * FROM train")
        assembler = VectorAssembler(
            inputCols=trainDF.columns[1:],
            outputCol="features"
        )
        model_training = assembler.transform(trainDF)
        model_training.createOrReplaceTempView("model_training")
        modelDataDF = spark.sql("SELECT a.row_num as id, a.yvals as label, b.features FROM yvals a LEFT JOIN model_training b ON a.row_num = b.row_num")
        model_training.createOrReplaceTempView("model_train")
        testDF = spark.sql("SELECT * FROM test")
        model_testing = assembler.transform(testDF)
        model_testing.createOrRepalceTempView("model_testing")
        model_test = spark.sql("SELECT row_num as id, features from model_testing")
        model_test.createOrReplaceTempView("model_test")
    # run
    gen_assembly()


# # get columns with _cat in name
# cat_cols = list(filter(lambda nm: nm.find("_cat") != -1, trainDF.columns))
# updated_cols = list(filter(lambda nm: nm.find("_cat") == -1, trainDF.columns))
# for c in cat_cols:
#     encoder = OneHotEncoder(inputCol=c, outputCol = c + '_vec')
#     onehotdata = encoder.fit(trainDF).transform(trainDF)
#     onehotdata.show(10, truncate=False)






# testDF = spark.sql("SELECT * FROM test")
# testoutdata = assembler.transform(testDF)
# testoutdata.createOrReplaceTempView("testout")
# testDF = spark.sql("SELECT row_num as id, features from testout")
#
# lr = LogisticRegression(maxIter=10)
# pipeline = Pipeline(stages=[lr])
#
# paramGrid = ParamGridBuilder() \
#     .addGrid(lr.regParam, [0.1, 0.01]) \
#     .build()
#
# modelDataDF = spark.sql("SELECT a.row_num as id, a.yvals as label, b.features FROM yvals a LEFT JOIN out b ON a.row_num = b.row_num")
#
# crossval = CrossValidator(estimator=pipeline,
#                           estimatorParamMaps=paramGrid,
#                           evaluator=BinaryClassificationEvaluator(),
#                           numFolds=2)
#
# cvModel = crossval.fit(modelDataDF)
#
# # Make predictions on test documents. cvModel uses the best model found (lrModel).
# prediction = cvModel.transform(testDF)
# selected = prediction.select("id", "features", "probability", "prediction")
# for row in selected.collect():
#     print(row)
