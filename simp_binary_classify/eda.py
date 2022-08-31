from simp_binary_classify.io import get_data
import pandas as pd
from plotnine import ggplot, geom_density, geom_histogram, aes, facet_wrap, ggtitle
from typing import List, Set
from scipy.stats import shapiro
from sys import float_info
import math
import statistics
from simp_binary_classify.spark import spark, sc
from pyspark.sql.types import Row
import os
from dotenv import load_dotenv

load_dotenv()


class EDA:
    # class variables
    train_data = get_data("train")
    test_data = get_data("test")
    col_nms = test_data.columns
    transform_mismatches = {}
    # initialize our trainsformed dataset
    train_range = range(train_data.shape[0])
    new_train_DF = (
        sc.parallelize([Row(x) for x in train_range])
        .toDF()
        .withColumnRenamed("_1", "row_num")
    )
    new_train_DF.createOrReplaceTempView("train")
    test_range = range(test_data.shape[0])
    new_test_DF = (
        sc.parallelize([Row(x) for x in test_range])
        .toDF()
        .withColumnRenamed("_1", "row_num")
    )
    new_test_DF.createOrReplaceTempView("test")
    # get environmental variables for testing
    cat_trunc_thres = float(os.getenv("category_trunc_threshold", "0.0"))
    unbal_thres = float(os.getenv("unbalanced_threshold", "10"))
    # instance variables
    def __init__(self):
        print("starting expoloratory data analysis")

    # function to check if variable is likely categorical
    def check_categorical(self, data: Set):
        # get decimal length
        def get_decimal_len(s: str) -> int:
            tmp = s.split(".")
            match len(tmp):
                case 1:
                    out = 0
                case other:
                    out = len(tmp[1])
            return out

        # check length of decimals
        dec_check = [get_decimal_len(str(x)) for x in data]
        return max(dec_check) < 2

    # need to shift values for log-normal transform
    def shift_log(self, data: List) -> List:
        shift = max([-x for x in data])
        return [math.log(x + shift + float_info.min) for x in data]

    # use shapiro-wilk test to test normal or log-normal
    def test_normal_or_log(self, data: List) -> (str, List):
        # run shapiro-wilk
        s_test = shapiro(data).statistic
        data_shifted = self.shift_log(data)
        s_test_log = shapiro(data_shifted).statistic
        match (s_test > s_test_log):
            case True:
                out = ("normal", data)
            case False:
                out = ("log-normal", data_shifted)
        return out

    # get_zscores
    def get_zscores(self, data: List) -> List:
        mu = statistics.mean(data)
        sd = statistics.stdev(data)
        return [(x - mu) / sd for x in data]

    # add new dataframe to new model data
    def add_new_DF(self, df: pd.DataFrame, name: str):
        c_nms = df.columns[1:]
        tmp_DF = spark.createDataFrame(df)
        tmp_DF.createOrReplaceTempView("tmp")
        b_col_nms = (
            str(list(map(lambda x: "b." + x, c_nms)))
            .strip("]")
            .strip("[")
            .replace("'", "")
        )
        updated_DF = spark.sql(
            "SELECT a.*, {} FROM {} a LEFT JOIN tmp b ON a.row_num = b.row_num".format(
                b_col_nms, name
            )
        )
        updated_DF.createOrReplaceTempView(name)

    # generate Distribution plot of normal or log-normal
    def gen_normal_dist(
        self, train_tup: (str, List), test_tup: (str, List), var_nm: str, used_str: str
    ) -> str:
        # create plot data
        test_nm = ["test" for i in range(len(test_tup[1]))]
        train_nm = ["train" for i in range(len(train_tup[1]))]
        comb_nm = ["combined" for i in range(len(test_tup[1]) + len(train_tup[1]))]
        comb_list = test_tup[1] + train_tup[1]
        pdat = pd.DataFrame(
            {
                "name": test_nm + train_nm + comb_nm,
                "data": test_tup[1] + train_tup[1] + comb_list,
            }
        )
        # plot
        plot_nm = var_nm + " " + train_tup[0] + " - " + used_str
        p = (
            (ggplot(pdat, aes("data", fill="factor(name)")))
            + geom_density()
            + facet_wrap("~name", scales="free_y", ncol=1)
            + ggtitle(plot_nm)
        )
        p.save("graphics/" + plot_nm + ".png")
        # add to spark model df if used
        add_var_nm = var_nm + "_" + "LN" if train_tup[0] == "log-normal" else "N"
        match used_str:
            case "Used":
                print("adding " + add_var_nm)
                train_addition_df = pd.DataFrame(
                    {"row_num": [i for i in self.train_range], add_var_nm: train_tup[1]}
                )
                self.add_new_DF(train_addition_df, "train")
                test_addition_df = pd.DataFrame(
                    {"row_num": [i for i in self.test_range], add_var_nm: test_tup[1]}
                )
                self.add_new_DF(test_addition_df, "test")
            case other:
                print("transforming " + add_var_nm)
        return plot_nm

    # generate state from zscore
    def gen_state_fromz(self, data: List, thres: float) -> List:
        def gen_state(d, thres) -> int:
            match (abs(d) >= thres):
                case True:
                    out = math.copysign(1, d)
                case False:
                    out = 0
            return out

        # create state list
        return [gen_state(x, thres) for x in data]

    # flip state and get zero values
    def flip_zero_one(self, s: int, d) -> int:
        match s:
            case 0:
                out = d
            case other:
                out = 0
        return out

    # binning and state variables
    def gen_binned_dist(
        self,
        train_tup: (str, List),
        test_tup: (str, List),
        train_zscores: List,
        var_nm: str,
        used_str: str,
    ) -> str:
        # create state variable
        train_state = self.gen_state_fromz(train_zscores, 1.645)
        test_zscores = self.get_zscores(test_tup[1])
        test_state = self.gen_state_fromz(test_zscores, 1.645)
        # create plot data
        test_nm = ["test" for i in range(len(test_tup[1]))]
        train_nm = ["train" for i in range(len(train_tup[1]))]
        comb_nm = ["combined" for i in range(len(test_tup[1]) + len(train_tup[1]))]
        comb_state = test_state + train_state
        pdat = pd.DataFrame(
            {
                "name": test_nm + train_nm + comb_nm,
                "data": test_state + train_state + comb_state,
            }
        )
        # plot binned state
        plot_nm = var_nm + " " + "binned state" + " - " + "Used"
        p = (
            (ggplot(pdat, aes("data", fill="factor(name)")))
            + geom_histogram(bins=len(set(train_state)))
            + facet_wrap("~name", scales="free_y", ncol=1)
            + ggtitle(plot_nm)
        )
        p.save("graphics/" + plot_nm + ".png")
        # plot interaction variable
        test_flip = [
            self.flip_zero_one(test_state[i], test_tup[1][i])
            for i in range(len(test_state))
        ]
        train_flip = [
            self.flip_zero_one(train_state[i], train_tup[1][i])
            for i in range(len(train_state))
        ]
        comb_flip = test_flip + train_flip
        pdat2 = pd.DataFrame(
            {"name": pdat.name, "data": test_flip + train_flip + comb_flip}
        )
        # plot state interaction variable
        plot_nm2 = var_nm + " " + "state interaction" + " - " + "Used"
        p = (
            (ggplot(pdat2, aes("data", fill="factor(name)")))
            + geom_density()
            + facet_wrap("~name", scales="free_y", ncol=1)
            + ggtitle(plot_nm2)
        )
        p.save("graphics/" + plot_nm2 + ".png")
        # add state and state variable interaction data
        # state
        add_var_nm = var_nm + "_" + "state"
        train_addition_df = pd.DataFrame(
            {"row_num": [i for i in self.train_range], add_var_nm: train_state}
        )
        self.add_new_DF(train_addition_df, "train")
        test_addition_df = pd.DataFrame(
            {"row_num": [i for i in self.test_range], add_var_nm: test_state}
        )
        self.add_new_DF(test_addition_df, "test")
        # state interaction
        add_var_nm = var_nm + "_" + "st_inter"
        train_addition_df = pd.DataFrame(
            {"row_num": [i for i in self.train_range], add_var_nm: train_flip}
        )
        self.add_new_DF(train_addition_df, "train")
        test_addition_df = pd.DataFrame(
            {"row_num": [i for i in self.test_range], add_var_nm: test_flip}
        )
        self.add_new_DF(test_addition_df, "test")
        return plot_nm

    # check how unbalanced categorical data is
    # this setup will only work with this data... but could be generalized
    def check_balance(self, data: List) -> float:
        # is less than
        def is_lt(d, thres):
            match (d <= thres):
                case True:
                    return 1
                case False:
                    return 0

        # is not equal
        def is_ne(d, tgt):
            match (d != tgt):
                case True:
                    return 1
                case False:
                    return 0

        num = sum([is_lt(x, 0) for x in data])
        denom = sum([is_ne(x, 0) for x in data])
        # output
        return num / denom

    # check if categories should be truncated
    def check_trunc(self, data: List, df: pd.DataFrame, cnm: str):
        # sum if data is equal to
        def is_eq_sum(d, tgt):
            match (d == tgt):
                case True:
                    return 1
                case False:
                    return 0
        # return original or replacement value
        def orig_or_replace(val, repl_val, check_list):
            match (val in check_list):
                case True:
                    return repl_val
                case False:
                    return val
        # find unique data set
        sorted_list = list(set(data))
        sorted_list.sort()
        # pull out test set to ensure all categories are available
        test_cat_check_data = set(df[df.name == "test"].data.to_list())
        test_cat_check = sum([x not in sorted_list for x in test_cat_check_data])
        match test_cat_check:
            case 0:
                print("all categories from test in train")
            case other:
                self.transform_mismatches.update(
                    {cnm: "test has categories not in training"}
                )
        # find a tuple of category and count
        cat_count = [(u, sum([is_eq_sum(x, u) for x in data])) for u in sorted_list]
        cmax = max([x[1] for x in cat_count])
        cat_ratios = [(x[0], round(x[1] / cmax, 4)) for x in cat_count]
        to_trunc = list(
            map(
                lambda x: x[0],
                filter(lambda x: x[1] <= self.cat_trunc_thres, cat_ratios),
            )
        )
        # if values exist to trunc, return updated df or otherwise original
        match len(to_trunc):
            case 0:
                return (False, df)
            case other:
                rp_val = statistics.mean(to_trunc)
                new_df = pd.DataFrame(
                    {
                        "name": df.name.to_list(),
                        "data": [
                            orig_or_replace(x, rp_val, to_trunc)
                            for x in df.data.to_list()
                        ],
                    }
                )
                return (True, new_df)
        return out

    # categorical histogram plot
    def gen_hist(self, df: pd.DataFrame, plot_nm: str):
        p = (
            (ggplot(df, aes("data", fill="factor(name)")))
            + geom_histogram(bins=len(df.data.unique()))
            + facet_wrap("~name", scales="free_y", ncol=1)
            + ggtitle(plot_nm)
        )
        p.save("graphics/" + plot_nm + ".png")
        # add transformed variable
        match plot_nm.find("Used"):
            case -1:
                print("only plotting for reference")
            case other:
                add_var_nm = plot_nm.replace(" Used", "")
                train_addition_df = pd.DataFrame(
                    {
                        "row_num": [i for i in self.train_range],
                        add_var_nm: df[df.name == "train"].data,
                    }
                )
                self.add_new_DF(train_addition_df, "train")
                test_addition_df = pd.DataFrame(
                    {
                        "row_num": [i for i in self.test_range],
                        add_var_nm: df[df.name == "test"].data,
                    }
                )
                self.add_new_DF(test_addition_df, "test")

    # plot Distributions
    def plot_dist(self, cnm: str):
        # is less than
        def is_gt(d, thres):
            match (d > thres):
                case True:
                    return 1
                case False:
                    return d

        # subset data
        test_list = self.test_data.loc[:, cnm].to_list()
        train_list = self.train_data.loc[:, cnm].to_list()
        comb_list = test_list + train_list
        # check if categorical
        train_set = set(train_list)
        cat_check = self.check_categorical(train_set)
        match cat_check:
            case True:
                # create plot data for categorical
                test_nm = ["test" for i in range(len(test_list))]
                train_nm = ["train" for i in range(len(train_list))]
                comb_nm = ["combined" for i in range(len(test_list) + len(train_list))]
                pdat = pd.DataFrame(
                    {
                        "name": test_nm + train_nm + comb_nm,
                        "data": test_list + train_list + comb_list,
                    }
                )
                # check the balance of the data
                bal_check = self.check_balance(train_list)
                trunc_check = self.check_trunc(train_list, pdat, cnm)
                # address balance and truncation cases
                # balancing checking around 10... a nice round number
                match (bal_check > self.unbal_thres):
                    case True:
                        pdat2 = pd.DataFrame(
                            {
                                "name": pdat.name,
                                "data": [is_gt(x, 0) for x in pdat.data.to_list()],
                            }
                        )
                        # plot
                        self.gen_hist(pdat2, cnm + "_binarized" + " Used")
                        self.gen_hist(
                            pdat, cnm + "_categorical" + " For Reference Only"
                        )
                    case False:
                        # check for truncation case
                        match trunc_check[0]:
                            case True:
                                # plot truncation
                                self.gen_hist(
                                    trunc_check[1],
                                    cnm + "_trunc_cat" + " Used",
                                )
                                # checking
                                print("CHECKING TRUNCATION")
                                print(trunc_check[1].data.unique())
                                # and original for reference
                                self.gen_hist(pdat, cnm + "_cat" + " For Reference Only")
                            case False:
                                # plot original category ohly
                                self.gen_hist(pdat, cnm + "_cat" + " Used")
            case False:
                # not categorical # check normality
                # normal transform to use -- use combined
                comb_tup = self.test_normal_or_log(comb_list)
                train_tup = self.test_normal_or_log(train_list)
                test_tup = self.test_normal_or_log(test_list)
                # catching mismatch
                if train_tup[0] == test_tup[0]:
                    mismatch_str = (
                        "training set suggests {}, while test set suggests {}".format(
                            train_tup[0], test_tup[0]
                        )
                    )
                    self.transform_mismatches.update({cnm: mismatch_str})
                    train_tup_input = train_tup
                    test_tup_input = test_tup
                    zscores = self.get_zscores(train_tup[1])
                else:
                    match comb_tup[0]:
                        case "normal":
                            train_tup_input = ("normal", train_list)
                            test_tup_input = ("normal", test_list)
                            zscores = self.get_zscores(train_tup_input[1])
                            print("making both transforms normal for " + cnm)
                        case "log-normal":
                            train_tup_input = ("log-normal", self.shift_log(train_list))
                            test_tup_input = ("log-normal", self.shift_log(test_list))
                            zscores = self.get_zscores(train_tup_input[1])
                            print("making both transforms log-normal for " + cnm)

                match ((max([abs(x) for x in zscores])) > 1.645):
                    case False:
                        self.gen_normal_dist(train_tup, test_tup, cnm, "Used")
                    case True:
                        self.gen_normal_dist(
                            train_tup_input, test_tup_input, cnm, "For Reference Only"
                        )
                        self.gen_binned_dist(
                            train_tup_input, test_tup_input, zscores, cnm, "Used"
                        )
