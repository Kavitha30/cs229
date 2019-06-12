'''
Global constants
'''

import datetime

DATA_DIR = 'data/'
PICKLES_DIR = DATA_DIR + 'pickled_data/'
DATA_SET_FILENAMES = ['xaa', 'xab', 'xac', 'xad', 'xae', 'xaf', 'xag', 'xah', 'xai', 'xaj', 'xak', 'xal']
CATEGORICAL_COLUMNS_ONE_HOT = ['verification_status', 'home_ownership', 'addr_state', 'purpose', 'grade']
CATEGORICAL_COLUMNS_BINARY = []
NORMALIZE_COLUMNS = ['loan_amnt' , 'funded_amnt',	'term',	'int_rate',	'installment',
                     'emp_length',	'annual_inc', 'dti', 'earliest_cr_line', 'revol_util']

DROP_COLUMNS = ['total_pymnt', 'recoveries']

LABELS = ['Fully Paid', 'Default', 'Charged Off']

#Multi Threading
NUM_WORKERS = 8

#Dates
START_DATE = datetime.datetime(2007, 1, 1)
END_DATE = datetime.datetime(2018, 12, 31)

#expansion data
UNEMPLOYMENT_PICKLE = PICKLES_DIR + 'state_to_unemployment.pickle'
HPI_PICKLE = PICKLES_DIR + 'state_to_hpi.pickle'
CPI_PICKLE = PICKLES_DIR + 'year_to_cpi.pickle'
SP500_PICKLE = PICKLES_DIR + 'year_to_sp500.pickle'
FFR_PICKLE = PICKLES_DIR + 'year_to_ffr.pickle'
INFLATION_PICKLE = PICKLES_DIR + 'year_to_inflation.pickle'
SPREAD_PICKLE = PICKLES_DIR + 'year_to_spread.pickle'
RM_RF_PICKLE = PICKLES_DIR + 'year_to_Rf-Rm.pickle'
SMB_PICKLE = PICKLES_DIR + 'year_to_smb.pickle'
HML_PICKLE = PICKLES_DIR + 'year_to_hml.pickle'
STATE_GDP = PICKLES_DIR + 'state_to_gdp.pickle'
COINCIDENT_EA = PICKLES_DIR + 'state_to_cea.pickle'

ROI_DICT = PICKLES_DIR + 'ROI_by_grade'
LOSS_DICT = PICKLES_DIR + 'Loss_by_grade'
LOSS_UNIFORM_DICT = PICKLES_DIR + 'Loss_uniform_by_grade'
PI_DICT = PICKLES_DIR + 'pi_by_grade'

