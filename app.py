from simp_binary_classify.eda import EDA
from simp_binary_classify.spark import spark
from simp_binary_classify.assembler import assemble_features
import sys

if __name__ == "__main__":
    # check version
    match sys.version.find('3.10'):
        case 0:
            # initial EDA
            eda = EDA()
            # loop over variables
            for nm in eda.col_nms:
                eda.gen_scatter(nm)
                eda.plot_dist(nm)

            # assemble the feature data sets
            assemble_features(eda.train_data)

            spark.sql("SELECT * FROM train").show()
        case other:
            print('Please install python 3.10')
