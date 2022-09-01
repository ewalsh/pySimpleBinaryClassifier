from pyspark.ml.feature import OneHotEncoder, VectorAssembler, StringIndexer, StandardScaler
import pandas as pd
from simp_binary_classify.spark import spark
import os
from dotenv import load_dotenv

load_dotenv()

def assemble_features(train_data):
    # get env details
    scale_data = os.getenv("scale_data", "False") == "True"
    cat_transform = os.getenv("cat_transform", "False") == "True"
    # create assembler function
    def gen_assembly():
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
        #
        # if cat transform is true, transform all cat variables
        if cat_transform:
            print("cat transform in assembler")
            trainDF = spark.sql("SELECT * FROM train")
            str_indexer = StringIndexer()
            cat_cols = list(filter(lambda x: x.find('cat') != -1, trainDF.columns[1:]))
            str_indexer.setInputCols(cat_cols)
            cat_cols_ind = list(map(lambda x: x + '_Index', cat_cols))
            str_indexer.setOutputCols(cat_cols_ind)
            updated_ind = str_indexer.fit(trainDF)
            updated_df = updated_ind.transform(trainDF)
            ohe = OneHotEncoder()
            ohe.setInputCols(cat_cols_ind)
            ohe.setOutputCols(list(map(lambda x: x[:(len(x)-6)] + '_classVec', cat_cols_ind)))
            updated_ohe = ohe.fit(updated_df)
            updated_df2 = updated_ohe.transform(updated_df)
            updated_df2.createOrReplaceTempView("train")
            # test dataset
        if scale_data:
            print("scale data in assembler")
            # replace df in case of transform update
            trainDF = spark.sql("SELECT * FROM train")
            assembler = VectorAssembler(
                inputCols=trainDF.columns[1:],
                outputCol="vectorized_features"
            )
            model_training = assembler.transform(trainDF)
            scaler = StandardScaler()
            scaler.setInputCol("vectorized_features")
            scaler.setOutputCol("features")
            scaler_model = scaler.fit(model_training)
            updated_df = scaler_model.transform(model_training)
            updated_df.createOrReplaceTempView("model_training")
        else:
            print("assembling without scaling")
            # update df
            trainDF = spark.sql("SELECT * FROM train")
            assembler = VectorAssembler(
                inputCols=trainDF.columns[1:],
                outputCol="features"
            )
            model_training = assembler.transform(trainDF)
            model_training.createOrReplaceTempView("model_training")
        # collate
        modelDataDF = spark.sql("SELECT a.row_num as id, a.yvals as label, b.features FROM yvals a LEFT JOIN model_training b ON a.row_num = b.row_num")
        modelDataDF.createOrReplaceTempView("model_train")
        testDF = spark.sql("SELECT * FROM test")
        model_testing = assembler.transform(testDF)
        model_testing.createOrReplaceTempView("model_testing")
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
