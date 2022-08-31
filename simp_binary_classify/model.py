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

def gen_model(model_train: pyspark.sql.DataFrame, num_splits: int):
    lr = LogisticRegression()
    grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 1]).build()
    evaluator = BinaryClassificationEvaluator()
    tvs = TrainValidationSplit(estimator=lr,\
                               estimatorParamMaps=grid,\
                               evaluator=evaluator,\
                               parallelism=cpu_count(),\
                               seed=29)
    #
    model_train = spark.sql("SELECT * FROM model_train")
    tvsModel = tvs.fit(model_train)
    tvsModel.getTrainRatio()
    tvsModel.validationMetrics
