from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.tuning import TrainValidationSplitModel, Param
from multiprocessing import cpu_count
from simp_binary_classify.spark import spark

from pyspark.mllib.classification import SVMModel
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint

import os
from dotenv import load_dotenv

load_dotenv()

def sparse_to_labeledpt(label, features):
    tmp = [0 for x in range(x.features.size)]
    counter = 0
    for i in x.features.indices:
            tmp[i] = x.features.values[counter]
            counter = counter + 1
    return(LabeledPoint(float(x.label), tmp))

def features_to_list(features):
    tmp = [0 for x in range(x.features.size)]
    counter = 0
    for i in x.features.indices:
            tmp[i] = x.features.values[counter]
            counter = counter + 1
    return(tmp)

def gen_model(model_train: pyspark.sql.DataFrame, model_name: String, num_splits: int = 3):
    nfolds = float(os.getenv("nfolds", "10"))
    tv_split = float(os.getenv("test_validate_split", "0.75"))
    match model_name:
        case "logistic":
            lr = LogisticRegression()
            pipeline = Pipeline(stages=[lr])
            paramGrid = ParamGridBuilder() \
            .addGrid(lr.regParam,
                     [x * 0.01 for x in range(0, 10, 2)] +\
                     [x * 0.1 for x in range(1, 10)]) \
            .build()
            # grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 1]).build()
            evaluator = BinaryClassificationEvaluator()
            # tvs = TrainValidationSplit(estimator=lr,\
            #                            estimatorParamMaps=grid,\
            #                            evaluator=evaluator,\
            #                            parallelism=cpu_count(),\
            #                            seed=29)
            #
            cnt = model_train.count() * tv_split
            crossval = CrossValidator(estimator=pipeline,
                                      estimatorParamMaps=paramGrid,
                                      evaluator=BinaryClassificationEvaluator(),
                                      numFolds=nfolds)

            # for each num_split, randomly split data to create
            # a validation subgroup
            mod_list = []
            mse_list = []
            for i in range(num_splits):
                sub_sample_DF = spark.sql("SELECT * FROM model_train TABLESAMPLE (75 PERCENT)")
                sub_sample_DF.createOrReplaceTempView("sub")
                tgt_sample_DF = spark.sql("SELECT * FROM model_train WHERE id NOT IN (SELECT id FROM sub)")
                cvModel = crossval.fit(sub_sample_DF)
                sub_test = cvModel.transform(tgt_sample_DF)
                sub_test.createOrReplaceTempView("sub_test")
                oos_mse = spark.sql("SELECT SUM(pow(label - prediction, 2)) as mse FROM sub_test").collect()
                mse_list.append(oos_mse)
                mod_list.append(cvModel)
        case "svm":
            mod_list = []
            mse_list = []
            for i in range(num_splits):
                sub_sample_DF = spark.sql("SELECT * FROM model_train TABLESAMPLE (75 PERCENT)")
                sub_sample_DF.createOrReplaceTempView("sub")
                tgt_sample_DF = spark.sql("SELECT * FROM model_train WHERE id NOT IN (SELECT id FROM sub)")
                sub_model_train_rdd = sub_sample_DF.rdd.map(lambda x: sparse_to_labeledpt(x.label, x.features))
                svm = SVMWithSGD.train(sub_model_train_rdd, iterations=nfolds)
                tgt_rdd = tgt_sample_DF.rdd.map(lambda x: features_to_list(x.features))
    # model_train = spark.sql("SELECT * FROM model_train")
    # tvsModel = tvs.fit(model_train)
    # tvsModel.getTrainRatio()
    # tvsModel.validationMetrics
