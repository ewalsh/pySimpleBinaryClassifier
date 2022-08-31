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

def gen_model(model_train: pyspark.sql.DataFrame, model_name: String, num_splits: int = 3):
    nfolds = float(os.getenv("nfolds", "10"))
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
            cnt = model_train.count() * 0.75
            crossval = CrossValidator(estimator=pipeline,
                                      estimatorParamMaps=paramGrid,
                                      evaluator=BinaryClassificationEvaluator(),
                                      numFolds=nfolds)

            # for each num_split, randomly split data to create
            # a validation subgroup
            for i in range(num_splits):
                sub_sample_DF = spark.sql("SELECT * FROM model_train TABLESAMPLE (75 PERCENT)")
                sub_sample_DF.createOrReplaceTempView("sub")
                tgt_sample_DF = spark.sql("SELECT * FROM model_train WHERE id NOT IN (SELECT id FROM sub)")
                cvModel = crossval.fit(sub_sample_DF)
        case "svm":
            model_train_rdd = model_train.rdd.map(lambda x: LabeledPoint(float(x.label), x.features.values))

    # model_train = spark.sql("SELECT * FROM model_train")
    # tvsModel = tvs.fit(model_train)
    # tvsModel.getTrainRatio()
    # tvsModel.validationMetrics
