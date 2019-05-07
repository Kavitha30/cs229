import src.util as util
import constants as c
import pandas as pd

def main(dataset):

    # add multi-hot columns for categories
    for category in c.CATEGORICAL_COLUMNS_ONE_HOT:
        print('transforming ' + category)
        new_cols = util.cat_to_array(dataset, category)
        dataset = pd.concat([dataset, new_cols], axis=1)

    for category in c.CATEGORICAL_COLUMNS_BINARY:
        print('transforming ' + category)
        new_cols = util.binary_encode(dataset, category)
        dataset = pd.concat([dataset, new_cols], axis=1)
    return dataset
    # dataset.to_pickle(c.DATA_DIR + 'full_data_clean.pkl')