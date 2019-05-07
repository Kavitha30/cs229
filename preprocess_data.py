import os
import pandas as pd
import src.category_encoding as category_encoding
import src.util as util
from src.data_expansion import generate as generate_expansion

import constants as c

DATA_DIR = './data/'

# IMPORTANT VARIABLE:
# Setting to True will generate ALL .csv files for each step, including the ones that we already have generated
# Setting to False improves efficiency by only generating missing data sets.

# Expansion pickle files
GENERATE_EXPANSION = True  # If False we can ignore the rest
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

def main():
    '''
    Running this script will go through all steps to generate the final dataset.
    The steps are described in the 'steps' list.
    It will first check for datasets in a backwards fashion so we download only the dataset we need to keep going.
    For example, if we already have the dataset for step 2, it will load it, and continue to step 3, it will not even
    check for step 1, assuming that step 2 built off of step 1. Note that we can force all steps bu setting the
    FORCE_GENERATION flag to True.
    '''
    if(os.path.isfile(c.DATA_DIR + 'full_data.pkl')):
        original_dataset = pd.read_pickle(c.DATA_DIR + 'full_data.pkl')
    else:
        original_dataset = util.load_raw_data()

    if GENERATE_EXPANSION:
        print('Generating expansion dataset.')
        generate_expansion(original_dataset,
                           generate_unemployment=GENERATE_UNEMPLOYMENT,
                           generate_hpi=GENERATE_HPI, generate_cpi=GENERATE_CPI, generate_sp500=GENERATE_SP500,
                           generate_inf=GENERATE_INF, generate_ffr=GENERATE_FFR, generate_fama_french=GENERATE_FAMA_FRENCH,
                           generate_spread=GENERATE_SPREAD, generate_state_gdp=GENERATE_STATE_GDP)

    #TODO: merge expansion dictionaries with dataset

    if CATEGORY_ENCODING:
        category_encoding.main(original_dataset)
        #TODO: drop original categories


if __name__ == '__main__':
    main()