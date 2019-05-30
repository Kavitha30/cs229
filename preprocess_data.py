import os
import pandas as pd
import src.category_encoding as category_encoding
import src.util as util
from src.data_expansion import generate as generate_expansion
from src.merge_expansion import merge as merge_expansion
import src.split_by_year as split_by_year

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

CATEGORY_ENCODING = False

SPLIT_BY_YEAR = False
sample = False

MERGE_EXPANSION = True
FINALIZE_DATA = True

def main():
    '''
    Running this script will go through all steps to generate the final dataset.
    '''

    # if(os.path.isfile(c.DATA_DIR + 'full_data.pkl')):
    #     dataset = pd.read_pickle(c.DATA_DIR + 'full_data.pkl')
    #     #dataset = dataset.tail(100000)
    # else:
    #     dataset = util.load_raw_data()

    if sample:
        if os.path.isfile(c.DATA_DIR + 'random_sample_50_data.pkl'):
            dataset = pd.read_pickle(c.DATA_DIR + 'random_sample_50_data.pkl')
        else:
            dataset = util.load_raw_data().sample(50)

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
        if os.path.isfile(c.DATA_DIR + 'full_data_clean.txt'):
            dataset = pd.read_csv(c.DATA_DIR + 'full_data_clean.txt', sep = '\t')
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
        dataset.loan_status = dataset.loan_status.replace({"Current": "0", "Fully Paid": "0", "Late (31-120 days)": "0",
                                                           "Default": "1", "Charged Off": "1", "In Grace Period": "0",
                                                           "Late (31-120 days)": "0",
                                                           "Does not meet the credit policy. Status:Fully Paid": "0",
                                                           "Does not meet the credit policy. Status:Charged Off": "0",
                                                           "Late (16-30 days)": "0"})
        # print(dataset['loan_status'].unique())
        print('Saving...')
        dataset.to_csv(path_or_buf=c.DATA_DIR + 'complete_dataset_expanded.txt', sep="\t",
                       encoding='utf-8', index=False)


if __name__ == '__main__':
    main()