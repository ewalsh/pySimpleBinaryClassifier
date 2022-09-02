from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.tuning import TrainValidationSplitModel, Param
from multiprocessing import cpu_count
from simp_binary_classify.spark import spark, sc
import statistics
import pandas as pd
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.classification import BinaryLogisticRegressionSummary
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os
import csv
from dotenv import load_dotenv

load_dotenv()

def sparse_to_labeledpt(label, features):
    tmp = [0 for x in range(features.size)]
    if str(type(features)) =="<class 'pyspark.ml.linalg.DenseVector'>":
        ind = list(range(0, features.size))
    else:
        ind = features.indices
    counter = 0
    for i in ind:
        tmp[i] = features.values[counter]
        counter = counter + 1
    return(LabeledPoint(float(label), tmp))

def features_to_list(features):
    tmp = [0 for x in range(features.size)]
    if str(type(features)) =="<class 'pyspark.ml.linalg.DenseVector'>":
        ind = list(range(0, features.size))
    else:
        ind = features.indices
    counter = 0
    for i in ind:
        tmp[i] = features.values[counter]
        counter = counter + 1
    return(tmp)

def check_best_metrics(metric_dict):
    try:
        with open('./metrics/best_validated.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                current_best = {'mroc': row['mroc'], 'sdroc': row['sdroc'], 'name': row['name']}
        # check current agaist best
        return(round(float(current_best['mroc']),2) <= round(float(metric_dict['mroc']), 2))
    except:
        print("no best_validated file in metrics folder")

def gen_model(model_train, num_splits: int = 3):
    # collect parameters for modelling and saving
    category_trunc_threshold = float(os.getenv("category_trunc_threshold", "0.01"))
    unbalanced_threshold = float(os.getenv("unbalanced_threshold", "10"))
    category_convert = (os.getenv("category_convert", "False")) == "True"
    cat_transform = (os.getenv("cat_transform", "False")) == "True"
    scale_data = (os.getenv("scale_data", "False")) == "True"
    nfolds = float(os.getenv("nfolds", "10"))
    tv_split = float(os.getenv("test_validate_split", "0.75"))
    gbt_depth = float(os.getenv("gbt_depth", "45"))
    gbt_learn = float(os.getenv("gbpt_learn_rt", "0.05"))
    model = os.getenv("model", "lr")
    match model:
        case "lr":
            lr = LogisticRegression()
            pipeline = Pipeline(stages=[lr])
            paramGrid = ParamGridBuilder() \
            .addGrid(lr.regParam,
                     [x * 0.01 for x in range(0, 10, 2)] +\
                     [x * 0.1 for x in range(1, 10)]) \
            .build()
            evaluator = BinaryClassificationEvaluator()
            cnt = model_train.count() * tv_split
            crossval = CrossValidator(estimator=pipeline,
                                      estimatorParamMaps=paramGrid,
                                      evaluator=BinaryClassificationEvaluator(),
                                      numFolds=nfolds)

            # for each num_split, randomly split data to create
            # a validation subgroup
            mod_list = []
            mse_list = []
            roc_list = []
            for i in range(num_splits):
                sub_sample_DF = spark.sql("SELECT * FROM model_train TABLESAMPLE (75 PERCENT)")
                sub_sample_DF.createOrReplaceTempView("sub")
                tgt_sample_DF = spark.sql("SELECT * FROM model_train WHERE id NOT IN (SELECT id FROM sub)")
                cvModel = crossval.fit(sub_sample_DF)
                sub_test = cvModel.transform(tgt_sample_DF)
                sub_test.createOrReplaceTempView("sub_test")
                sub_test_rows = sub_test.collect()
                oos_mse = spark.sql("SELECT SUM(pow(label - prediction, 2)) as mse FROM sub_test").collect()
                # tp = spark.sql("SELECT COUNT(*) FROM sub_test WHERE label = 1 AND prediction = label")
                # p = spark.sql("SELECT COUNT(*) FROM sub_test WHERE label = prediction")
                p = list(filter(lambda x: x.prediction == 1.0, sub_test_rows))
                tp = list(filter(lambda x: x.label == 1.0, p))
                tpr = len(tp)/len(p)
                fp = list(filter(lambda x: x.label == 0.0, p))
                n = list(filter(lambda x: x.prediction == 0.0, sub_test_rows))
                fpr = len(fp)/len(n)
                roc = tpr/fpr
                mse_list.append(oos_mse[0])
                mod_list.append(cvModel)
                roc_list.append(roc)
            # analyze mean mse and look at sd for stability
            rel_cnt = sub_test.count()
            rel_list = list(map(lambda x: x.mse / rel_cnt, mse_list))
            mmse = statistics.mean(rel_list)
            sdmse = statistics.stdev(rel_list)
            mroc = statistics.mean(roc_list)
            sdroc = statistics.stdev(roc_list)
            # read current best validated metrics
            # if superior run model on full test set and
            # make predictions on actual test data
            model_name = model + '_' + str(tv_split) + '_' + str(nfolds) +\
            '_' + str(category_convert) + '_' + str(cat_transform) +\
            '_' + str(scale_data) + '_' + str(unbalanced_threshold) +\
            '_' + str(category_trunc_threshold)
            validated_metrics = {'mroc': mroc, 'sdroc': sdroc, 'name': model_name}
            metrics_check = check_best_metrics(validated_metrics)
            if metrics_check:
                print("model is a new best!")
                # write new best metrics
                with open('./metrics/best_validated.csv', 'w', newline='') as f:
                    w = csv.DictWriter(f, validated_metrics.keys())
                    w.writeheader()
                    w.writerow(validated_metrics)
                # train model on full test set
                full_train_data = spark.sql("SELECT * FROM model_train")
                full_cvModel = crossval.fit(full_train_data)
                full_train = full_cvModel.transform(full_train_data)
                # plot ROC
                # find test model predictions
                full_test_data = spark.sql("SELECT * FROM model_test")
                full_test = full_cvModel.transform(full_test_data)
                full_test.createOrReplaceTempView("full_test")
                # collect test predictions
                preds_DF = spark.sql("SELECT id, prediction FROM full_test ORDER BY id")
                preds = preds_DF.collect()
                preds_df = pd.DataFrame(
                    {
                        'model_name': [model_name for x in preds],
                        'id': [x.id for x in preds],
                        'prediction': [x.prediction for x in preds]
                    }
                )
                preds_df.to_csv('./metrics/predictions.csv')
            else:
                print("model doesn't beat current best")
        case "svm":
            mod_list = []
            mse_list = []
            roc_list = []
            for i in range(num_splits):
                sub_sample_DF = spark.sql("SELECT * FROM model_train TABLESAMPLE (75 PERCENT)")
                sub_sample_DF.createOrReplaceTempView("sub")
                tgt_sample_DF = spark.sql("SELECT * FROM model_train WHERE id NOT IN (SELECT id FROM sub)")
                # sub_model_train_rdd = sub_sample_DF.rdd.map(lambda x: sparse_to_labeledpt(x.label, x.features))
                subRows = sub_sample_DF.collect()
                train_list = []
                for x in subRows:
                    # print(x.features)
                    train_list.append(sparse_to_labeledpt(x.label, x.features))
                svm = SVMWithSGD.train(sc.parallelize(train_list), iterations=nfolds*100)
                # tgt_rdd = tgt_sample_DF.rdd.map(lambda x: features_to_list(x.features))
                testRows = tgt_sample_DF.collect()
                test_label = []
                test_pred = []
                for x in testRows:
                    test_features = features_to_list(x.features)
                    test_pred.append(svm.predict(test_features))
                    test_label.append(x.label)
                # collect results
                svm_df = pd.DataFrame(
                    {
                        "prediction": test_pred,
                        "label": test_label
                    }
                )
                # collect metrics
                p = svm_df[svm_df["prediction"] == 1.0]
                tp = p[p["label"] == 1.0]
                tpr = tp.shape[0]/p.shape[0]
                fp = p[p["label"] == 0.0]
                n = svm_df[svm_df["prediction"] == 0.0]
                fpr = len(fp)/len(n)
                roc = tpr/fpr
                mod_list.append(svm)
                roc_list.append(roc)
            # analyze mean mse and look at sd for stability
            mroc = statistics.mean(roc_list)
            sdroc = statistics.stdev(roc_list)
            # read current best validated metrics
            # if superior run model on full test set and
            # make predictions on actual test data
            model_name = model + '_' + str(tv_split) + '_' + str(nfolds) +\
            '_' + str(category_convert) + '_' + str(cat_transform) +\
            '_' + str(scale_data) + '_' + str(unbalanced_threshold) +\
            '_' + str(category_trunc_threshold)
            validated_metrics = {'mroc': mroc, 'sdroc': sdroc, 'name': model_name}
            metrics_check = check_best_metrics(validated_metrics)
            if metrics_check:
                print("model is a new best!")
                # write new best metrics
                with open('./metrics/best_validated.csv', 'w', newline='') as f:
                    w = csv.DictWriter(f, validated_metrics.keys())
                    w.writeheader()
                    w.writerow(validated_metrics)
                # train model on full test set
                full_train_data = spark.sql("SELECT * FROM model_train")
                subRows = full_train_data.collect()
                train_list = []
                for x in subRows:
                    # print(x.features)
                    train_list.append(sparse_to_labeledpt(x.label, x.features))
                svm = SVMWithSGD.train(sc.parallelize(train_list), iterations=nfolds*100)
                train_label = []
                train_pred = []
                for x in subRows:
                    train_features = features_to_list(x.features)
                    train_pred.append(svm.predict(train_features))
                    train_label.append(x.label)
                # find test model predictions
                full_test_data = spark.sql("SELECT * FROM model_test")
                testRows = full_test_data.collect()
                test_label = []
                test_pred = []
                test_ids = []
                for x in testRows:
                    test_features = features_to_list(x.features)
                    test_pred.append(svm.predict(test_features))
                    test_label.append(x.label)
                    test_ids.append(x.id)
                # collect predcitions
                model_name = model + '_' + str(tv_split) + '_' + str(nfolds) +\
                '_' + str(category_convert) + '_' + str(cat_transform) +\
                '_' + str(scale_data) + '_' + str(unbalanced_threshold) +\
                '_' + str(category_trunc_threshold)
                preds_df = pd.DataFrame(
                    {
                        'model_name': [model_name for x in test_pred],
                        'id': test_ids,
                        'prediction': test_pred
                    }
                )
                preds_df.to_csv('./metrics/predictions.csv')
            else:
                print("model doesn't beat current best")
        case "gbt":
            # print("work in progress")
            gbt = GBTClassifier(labelCol = 'label', featuresCol = 'features')
            # gbt.setMaxIter(300)
            # gbt.setMaxDepth(gbt_depth)
            # gbt.setStepSize(gbt_learn)
            # for each num_split, randomly split data to create
            # a validation subgroup
            mod_list = []
            mse_list = []
            roc_list = []
            for i in range(num_splits):
                sub_sample_DF = spark.sql("SELECT * FROM model_train TABLESAMPLE (75 PERCENT)")
                sub_sample_DF.createOrReplaceTempView("sub")
                tgt_sample_DF = spark.sql("SELECT * FROM model_train WHERE id NOT IN (SELECT id FROM sub)")
                gbtModel = gbt.fit(sub_sample_DF)
                sub_test = gbtModel.transform(tgt_sample_DF)
                sub_test.createOrReplaceTempView("sub_test")
                sub_test_rows = sub_test.collect()
                oos_mse = spark.sql("SELECT SUM(pow(label - prediction, 2)) as mse FROM sub_test").collect()
                # tp = spark.sql("SELECT COUNT(*) FROM sub_test WHERE label = 1 AND prediction = label")
                # p = spark.sql("SELECT COUNT(*) FROM sub_test WHERE label = prediction")
                p = list(filter(lambda x: x.prediction == 1.0, sub_test_rows))
                tp = list(filter(lambda x: x.label == 1.0, p))
                tpr = len(tp)/len(p)
                fp = list(filter(lambda x: x.label == 0.0, p))
                n = list(filter(lambda x: x.prediction == 0.0, sub_test_rows))
                fpr = len(fp)/len(n)
                roc = tpr/fpr
                mse_list.append(oos_mse[0])
                mod_list.append(gbtModel)
                roc_list.append(roc)
            # analyze mean mse and look at sd for stability
            mroc = statistics.mean(roc_list)
            sdroc = statistics.stdev(roc_list)
            # read current best validated metrics
            # if superior run model on full test set and
            # make predictions on actual test data
            model_name = model + '_' + str(tv_split) + '_' + str(nfolds) +\
            '_' + str(category_convert) + '_' + str(cat_transform) +\
            '_' + str(scale_data) + '_' + str(unbalanced_threshold) +\
            '_' + str(category_trunc_threshold) +\
            '_' + str(gbt_depth) + '_' + str(gbt_learn)
            validated_metrics = {'mroc': mroc, 'sdroc': sdroc, 'name': model_name}
            metrics_check = check_best_metrics(validated_metrics)
            if metrics_check:
                print("model is a new best!")
                # write new best metrics
                with open('./metrics/best_validated.csv', 'w', newline='') as f:
                    w = csv.DictWriter(f, validated_metrics.keys())
                    w.writeheader()
                    w.writerow(validated_metrics)
                # train model on full test set
                full_train_data = spark.sql("SELECT * FROM model_train")
                full_gbtModel = gbt.fit(full_train_data)
                full_train = full_gbtModel.transform(full_train_data)
                # plot ROC
                # find test model predictions
                full_test_data = spark.sql("SELECT * FROM model_test")
                full_test = full_gbtModel.transform(full_test_data)
                full_test.createOrReplaceTempView("full_test")
                # collect test predictions
                preds_DF = spark.sql("SELECT id, prediction FROM full_test ORDER BY id")
                preds = preds_DF.collect()
                preds_df = pd.DataFrame(
                    {
                        'model_name': [model_name for x in preds],
                        'id': [x.id for x in preds],
                        'prediction': [x.prediction for x in preds]
                    }
                )
                preds_df.to_csv('./metrics/predictions.csv')
            else:
                print("model doesn't beat current best")
        case other:
            print("{} model hasn't been added yet!".format(model))
