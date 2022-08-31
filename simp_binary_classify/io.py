import pandas as pd
from typing import Optional

"""
Gets test data
@param data_type A string of either test or train to pull relevant data
@return an optional dataframe of data
"""


def get_data(data_type: str) -> Optional[pd.DataFrame]:
    match data_type:
        case "test":
            out = pd.read_csv(
                "https://econdatasci.s3.eu-west-2.amazonaws.com/data/public_test.csv"
            )
        case "train":
            out = pd.read_csv(
                "https://econdatasci.s3.eu-west-2.amazonaws.com/data/public_train.csv"
            )
        case other:
            out = None
    return out
