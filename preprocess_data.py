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
        #dataset = dataset.tail(100000)
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
        #ignoring years 2019 since we don't have the data
        dataset = dataset[dataset.year!= 2019.0]
        #dropping original categorial variables
        dataset = dataset.drop(columns=['sub_grade', 'verification_status', 'pymnt_plan', 'purpose', 'initial_list_status',
            'application_type','addr_state', 'home_ownership'])
        #encoding classification
        #print(dataset['loan_status'].unique())
        dataset.loan_status = dataset.loan_status.replace({"Current": "0", "Fully Paid":"0","Late (31-120 days)":"0",
                                                   "Default": "1", "Charged Off": "1", "In Grace Period": "0", "Late (31-120 days)":"0",
                                                   "Does not meet the credit policy. Status:Fully Paid": "0", 
                                                   "Does not meet the credit policy. Status:Charged Off":"0",
                                                   "Late (16-30 days)":"0"})
        #print(dataset['loan_status'].unique())
        dataset.to_csv(path_or_buf='/Users/ahn 1/Desktop/CS229/cs229/data/complete_dataset.txt',sep="\t",encoding='utf-8',index=False)


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