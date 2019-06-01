'''
Global constants
'''

import datetime

DATA_DIR = 'data/'
PICKLES_DIR = DATA_DIR + 'pickled_data/'
DATA_SET_FILENAMES = ['xaa', 'xab', 'xac', 'xad', 'xae', 'xaf', 'xag', 'xah', 'xai', 'xaj', 'xak', 'xal']
CATEGORICAL_COLUMNS_ONE_HOT = ['verification_status', 'pymnt_plan', 'initial_list_status',
                               'application_type', 'home_ownership', 'addr_state', 'purpose', 'sub_grade']
CATEGORICAL_COLUMNS_BINARY = []
NORMALIZE_COLUMNS = ['CPI', 'Coincident_economic_activity', 'Corperate_bond_spread',
                     'Federal_fund_rate', 'GDP', 'HML', 'HPI', 'Inflation','RF_RM',
                     'SMB', 'SP500', 'Unemployment', 'acc_now_delinq', 'acc_open_past_24mths',
                     'all_util', 'annual_inc', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util',
                     'delinq_2yrs', 'dti', 'earliest_cr_line', 'emp_length',
                     'funded_amnt', 'funded_amnt_inv', 'il_util', 'inq_fi', 'inq_last_12m',
                     'inq_last_6mths', 'installment', 'int_rate', 'issue_d', 'last_credit_pull_d',
                     'loan_amnt', 'max_bal_bc', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op',
                     'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_last_delinq',
                     'mths_since_last_record', 'mths_since_rcnt_il', 'mths_since_recent_bc',
                     'mths_since_recent_inq', 'num_accts_ever_120_pd', 'open_acc', 'open_acc_6m',
                     'open_act_il', 'open_il_12m', 'open_il_24m', 'open_rv_12m', 'open_rv_24m',
                     'out_prncp', 'out_prncp_inv', 'pct_tl_nvr_dlq', 'percent_bc_gt_75',
                     'pub_rec_bankruptcies', 'revol_bal', 'revol_util', 'term', 'tot_coll_amt',
                     'tot_cur_bal', 'tot_hi_cred_lim', 'total_acc', 'total_bal_ex_mort',
                     'total_bal_il', 'total_bc_limit', 'total_cu_tl', 'total_il_high_credit_limit',
                     'total_pymnt', 'total_pymnt_inv', 'total_rec_int', 'total_rec_prncp', 'total_rev_hi_lim']

DROP_COLUMNS = ['chargeoff_within_12_mths', 'year', 'Unnamed: 0', 'mths_since_last_delinq.1', 'index'] #'tot_coll_amt'?

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

