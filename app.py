from simp_binary_classify.eda import EDA
from simp_binary_classify.spark import spark
from simp_binary_classify.assembler import assemble_features
from simp_binary_classify.model import gen_model
import sys
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    # check version
    if sys.version.find('3.10') == 0 :
        # initial EDA
        eda = EDA()
        # loop over variables
        for nm in eda.col_nms:
            eda.gen_scatter(nm)
            eda.plot_dist(nm)

        # assemble the feature data sets
        assemble_features(eda.train_data)
        model_train = spark.sql("SELECT * FROM model_train")
        #
        num_splits = int(os.getenv("num_splits", "5"))
        model_name = os.getenv("model", "lr")
        # run model
        gen_model(model_train, num_splits)

        spark.sql("SELECT * FROM model_test").show()
    else:
        print('Please install python 3.10')
