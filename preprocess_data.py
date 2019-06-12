import os
import pandas as pd
import src.category_encoding as category_encoding
import src.util as util
from src.data_expansion import generate as generate_expansion
from src.merge_expansion import merge as merge_expansion
import src.split_by_year as split_by_year
import normalize as norm
from sklearn.model_selection import train_test_split
import numpy as np

import constants as c

DATA_DIR = './data/'

# IMPORTANT VARIABLE:
# Setting to True will generate ALL .csv files for each step, including the ones that we already have generated
# Setting to False improves efficiency by only generating missing data sets.

# Expansion pickle files
GENERATE_EXPANSION = False  # If False we can ignore the rest
GENERATE_UNEMPLOYMENT = False
GENERATE_HPI = False
GENERATE_CPI = True
GENERATE_MISERY = False
GENERATE_SP500 = True
GENERATE_INF = False
GENERATE_FFR = False
GENERATE_FAMA_FRENCH = True
GENERATE_SPREAD = False
GENERATE_STATE_GDP = True
GENERATE_CEA = False

CLEAN  = True
CATEGORY_ENCODING = True

SPLIT_BY_YEAR = False
sample = False

MERGE_EXPANSION = True
FINALIZE_DATA = True

Update = True


def main():
    '''
    Running this script will go through all steps to generate the final dataset.
    '''

    if(os.path.isfile(c.DATA_DIR + 'full_data.pkl')):
        dataset = pd.read_pickle(c.DATA_DIR + 'full_data.pkl')
    else:
        dataset = util.load_raw_data()

    if sample:
        if os.path.isfile(c.DATA_DIR + 'random_sample_50_data.pkl'):
            dataset = pd.read_csv(c.DATA_DIR + 'random_sample_50_data.pkl')
        else:
            dataset = util.load_raw_data().sample(50)


    if CLEAN:
        dataset = dataset[dataset.loan_status.isin(c.LABELS)]
        dataset = dataset.drop(c.DROP_COLUMNS, axis=1)


    if CATEGORY_ENCODING:
        print('Encoding categories')
        dataset = category_encoding.main(dataset)


    if SPLIT_BY_YEAR:
        print('Splitting data by year')
        dataset = split_by_year.main(dataset)
        #ignoring years 2019 since we don't have the data
        dataset = dataset[dataset.year!= 2019.0]
        dataset.to_csv(path_or_buf=c.DATA_DIR + 'full_data_clean.txt', sep="\t",
                       encoding='utf-8', index=False)


    if GENERATE_EXPANSION:
        print('Generating expansion dataset.')
        generate_expansion(dataset,
                           generate_unemployment=GENERATE_UNEMPLOYMENT,
                           generate_hpi=GENERATE_HPI, generate_cpi=GENERATE_CPI, generate_sp500=GENERATE_SP500,
                           generate_inf=GENERATE_INF, generate_ffr=GENERATE_FFR, generate_fama_french=GENERATE_FAMA_FRENCH,
                           generate_spread=GENERATE_SPREAD, generate_state_gdp=GENERATE_STATE_GDP, generate_cea=GENERATE_CEA)

    if MERGE_EXPANSION:
        print('Merging expansion data...')
        dataset = merge_expansion(dataset)

    if FINALIZE_DATA:
        # dropping original categorial variables
        print('Finalizing dataset...')
        dataset = dataset.drop(
            columns=['sub_grade', 'verification_status', 'pymnt_plan', 'purpose', 'initial_list_status',
                     'application_type', 'addr_state', 'home_ownership'])
        #fixing problematic values
        dataset = dataset[dataset.emp_length != 'na']
        dataset['emp_length'].replace(' 1 year', 1, inplace=True)
        dataset['emp_length'].replace('1 year', 1, inplace=True)
        # encoding classification
        # print(dataset['loan_status'].unique())
        dataset.loan_status = dataset.loan_status.replace({"Fully Paid": 0, "Default": 1, "Charged Off": 1})
        # print(dataset['loan_status'].unique())
        print('Splitting')
        X = dataset[dataset.columns.difference(['loan_status'])]
        y = dataset['loan_status'].values
        print(y)

        # test/val/train split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


        print('Normanlizing')
        X_train, X_test, X_val = norm.run(X_train, X_test, X_val)
        print('Saving...')
        x_to_write = [
            (X_train, 'x_train_exp.csv'),
            (X_test, 'x_test_exp.csv'),
            (X_val, 'x_val_exp.csv'),
        ]
        y_to_write = [
            (y_train, 'y_train_exp.csv'),
            (y_test, 'y_test_exp.csv'),
            (y_val, 'y_val_exp.csv'),
        ]
        for datasplit, filename in x_to_write:
            datasplit.to_csv(filename)
        for datasplit, filename in y_to_write:
            np.savetxt(filename, datasplit, fmt="%1i")
        print('data saved')

    if Update:
        X_train = pd.read_csv('x_train_exp.csv', sep=',')
        X_test = pd.read_csv('x_test_exp.csv', sep=',')
        X_val = pd.read_csv('x_val_exp.csv', sep=',')
        y_train = pd.read_csv('y_train_exp.csv', sep=',', header=None)
        y_test = pd.read_csv('y_test_exp.csv', sep=',', header=None)
        y_val = pd.read_csv('y_test_exp.csv', sep=',', header=None)


if __name__ == '__main__':
    main()