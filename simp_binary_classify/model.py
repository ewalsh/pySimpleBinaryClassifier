from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.tuning import TrainValidationSplitModel, Param
from multiprocessing import cpu_count

def gen_model(model_train: pyspark.sql.DataFrame):
    lr = LogisticRegression()
    grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 1]).build()
    evaluator = BinaryClassificationEvaluator()
    tvs = TrainValidationSplit(estimator=lr,\
                               estimatorParamMaps=grid,\
                               evaluator=evaluator,\
                               parallelism=cpu_count(),\
                               seed=29)
    #

tvsModel = tvs.fit(model_train)

tvsModel.getTrainRatio()
0.75

tvsModel.validationMetrics
