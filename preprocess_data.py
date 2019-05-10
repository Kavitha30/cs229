import os
import pandas as pd
import src.category_encoding as category_encoding
import src.util as util
from src.data_expansion import generate as generate_expansion
import src.split_by_year as split_by_year

import constants as c

DATA_DIR = './data/'

# IMPORTANT VARIABLE:
# Setting to True will generate ALL .csv files for each step, including the ones that we already have generated
# Setting to False improves efficiency by only generating missing data sets.

# Expansion pickle files
GENERATE_EXPANSION = False  # If False we can ignore the rest
GENERATE_UNEMPLOYMENT = True
GENERATE_HPI = True
GENERATE_CPI = False #TODO: get csv
GENERATE_MISERY = True
GENERATE_SP500=False #TODO: get csv
GENERATE_INF= True
GENERATE_FFR = True
GENERATE_FAMA_FRENCH = False #TODO: get csv
GENERATE_SPREAD = True
GENERATE_STATE_GDP = True

CATEGORY_ENCODING = True

SPLIT_BY_YEAR = True

sample = False

def main():
    '''
    Running this script will go through all steps to generate the final dataset.
    '''

    if(os.path.isfile(c.DATA_DIR + 'full_data.pkl')):
        dataset = pd.read_pickle(c.DATA_DIR + 'full_data.pkl')
    else:
        dataset = util.load_raw_data()

    if sample:
        if (os.path.isfile(c.DATA_DIR + 'random_sample_50_data.pkl')):
            dataset = pd.read_pickle(c.DATA_DIR + 'random_sample_50_data.pkl')
        else:
            dataset = util.load_raw_data().sample(50)

    if CATEGORY_ENCODING:
        print('Encoding categories')
        dataset = category_encoding.main(dataset)

    if SPLIT_BY_YEAR:
        print('Splitting data by year')
        dataset = split_by_year.main(dataset)
        print(dataset)
        # TODO: drop original categories (category columns are in constants.py)
        # TODO: Convert loan_status to label column (Current = still active, all other categories left as is: ChargeOff, Grace Period, Late, Paid in full, etc.)
        # TODO: Maybe drop all rows for 2019, (data only goes through february)
        dataset.to_pickle(c.DATA_DIR + 'full_data_clean.pkl')


    if GENERATE_EXPANSION:
        print('Generating expansion dataset.')
        generate_expansion(dataset,
                           generate_unemployment=GENERATE_UNEMPLOYMENT,
                           generate_hpi=GENERATE_HPI, generate_cpi=GENERATE_CPI, generate_sp500=GENERATE_SP500,
                           generate_inf=GENERATE_INF, generate_ffr=GENERATE_FFR, generate_fama_french=GENERATE_FAMA_FRENCH,
                           generate_spread=GENERATE_SPREAD, generate_state_gdp=GENERATE_STATE_GDP)

    #TODO: merge expansion dictionaries with dataset

if __name__ == '__main__':
    main()