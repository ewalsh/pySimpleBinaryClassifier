from simp_binary_classify.eda import EDA
# from multiprocessing import Process
from simp_binary_classify.spark import spark

if __name__ == "__main__":
    # initiate processes
    # processes = []
    # initial EDA
    eda = EDA()
    # loop over variables
    for nm in eda.col_nms:
        eda.plot_dist(nm)
        # p = Process(target=eda.plot_dist, args=(nm, ))
        # p.start()
        # processes.append(p)
        # eda = EDA()


    # for p in processes:
    #     p.join()

    spark.sql("SELECT * FROM train").show()

# def __main__():
    # from simp_binary_classify.eda import EDA
    # from multiprocessing import Process
# print("hello")
