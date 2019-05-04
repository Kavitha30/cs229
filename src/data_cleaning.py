import src.util as util
import constants as c
import pandas as pd

def main():
    util.load_raw_data()
    dataset = pd.read_pickle(c.DATA_DIR + 'full_data.pkl')

    # add multi-hot columns for categories
    for category in c.CATEGORICAL_COLUMNS_ONE_HOT:
        print('transforming ' + category)
        new_cols = util.cat_to_array(dataset, category)
        dataset.drop(category, axis = 1)
        dataset = pd.concat([dataset, new_cols], axis=1)

    for category in c.CATEGORICAL_COLUMNS_BINARY:
        print('transforming ' + category)
        new_cols = util.binary_encode(dataset, category)
        dataset.drop(category, axis = 1)
        dataset = pd.concat([dataset, new_cols], axis=1)

    dataset.to_pickle(c.DATA_DIR + 'full_data_clean.pkl')