from simp_binary_classify.eda import EDA
from simp_binary_classify.spark import spark
from simp_binary_classify.assembler import assemble_features

if __name__ == "__main__":
    # initial EDA
    eda = EDA()
    # loop over variables
    for nm in eda.col_nms:
        eda.gen_scatter(nm)
        eda.plot_dist(nm)

    # assemble the feature data sets
    assemble_features(eda.train_data)

    spark.sql("SELECT * FROM train").show()
