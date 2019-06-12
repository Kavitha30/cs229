import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime as dt
pd.set_option('display.max_columns', 500)

cols = [
    'loan_amnt',
    'funded_amnt',
    'funded_amnt_inv',
    'term',
    'int_rate',
    'installment',
    'grade',
    'sub_grade',
    'emp_length',
    'home_ownership',
    'annual_inc',
    'verification_status',
    'issue_d',
    'loan_status', # Y
    'purpose',
    'addr_state',
    'dti',
    'inq_last_6mths',
    'mths_since_last_delinq',
    'mths_since_last_record',
    'open_acc',
    'pub_rec',
    'total_acc',
    'initial_list_status',
    'total_pymnt', # Z
    'recoveries', # Z
    'mths_since_last_major_derog',
    'policy_code',
    'application_type',
    'annual_inc_joint',
    'dti_joint',
    'acc_now_delinq',
    'open_acc_6m',
    'open_act_il',
    'open_il_12m',
    'open_il_24m',
    'mths_since_rcnt_il',
    'open_rv_12m',
    'open_rv_24m',
    'all_util',
    'inq_fi',
    'inq_last_12m',
    'acc_open_past_24mths',
    'avg_cur_bal',
    'chargeoff_within_12_mths',
    'mo_sin_old_il_acct',
    'mo_sin_old_rev_tl_op',
    'mo_sin_rcnt_rev_tl_op',
    'mo_sin_rcnt_tl',
    'mort_acc',
    'mths_since_recent_bc',
    'mths_since_recent_bc_dlq',
    'mths_since_recent_inq',
    'mths_since_recent_revol_delinq',
    'num_accts_ever_120_pd',
    'num_actv_bc_tl',
    'num_actv_rev_tl',
    'num_bc_sats',
    'num_bc_tl',
    'num_il_tl',
    'num_op_rev_tl',
    'num_rev_accts',
    'num_rev_tl_bal_gt_0',
    'num_sats',
    'num_tl_120dpd_2m',
    'num_tl_30dpd',
    'num_tl_90g_dpd_24m',
    'num_tl_op_past_12m',
    'pct_tl_nvr_dlq',
    'percent_bc_gt_75',
    'pub_rec_bankruptcies',
    'tax_liens',
    'sec_app_inq_last_6mths',
    'sec_app_mort_acc',
    'sec_app_open_acc',
    'sec_app_revol_util',
    'sec_app_open_act_il',
    'sec_app_num_rev_accts',
    'sec_app_chargeoff_within_12_mths',
    'sec_app_collections_12_mths_ex_med',
    'sec_app_mths_since_last_major_derog',
]

percent_to_float = lambda s: 0. if s == '' else float(s[:-1])
converters = {
    'int_rate': percent_to_float,
}

truncated = False
filename = 'raw/trunc.csv' if truncated else 'raw/data.csv'
t = dt.now()
data = pd.read_csv(filename, usecols=cols, converters=converters)
print('read time', (dt.now() - t).total_seconds())
t = dt.now()

colnames = list(data.columns.values)
coltypes = data.dtypes
print(list(zip(colnames, coltypes)))

y0 = ['Fully Paid', 'Does not meet the credit policy. Status:Fully Paid']
y1 = ['Charged Off', 'Does not meet the credit policy. Status:Charged Off', 'Default']
data = data[data['loan_status'].isin(y0 + y1)]
n = len(data)

Y = np.zeros(n)
Y[data['loan_status'].isin(y1)] = 1
Z = data[['total_pymnt', 'recoveries']]
X = data.drop(labels=['loan_status', 'total_pymnt', 'recoveries'], axis=1)

X = pd.get_dummies(X, prefix_sep='=', dummy_na=True)
X.fillna(value=0, inplace=True)

colnames = list(X.columns.values)
coltypes = X.dtypes
print(list(zip(colnames, coltypes)))
print(len(colnames))
print(X.head())
print('n', n)
print('process time', (dt.now() - t).total_seconds())
t = dt.now()

if not truncated:
    split = [0, int(.8 * n), int(.9 * n), n]
    for i, name in enumerate(['train', 'val', 'test']):
        X_split = X[split[i] : split[i+1]]
        Y_split = Y[split[i] : split[i+1]]
        Z_split = Z[split[i] : split[i+1]]

        X_split.to_pickle('data/x_%s.pkl' % name)
        np.save('data/y_%s.npy' % name, Y_split)
        Z_split.to_pickle('data/z_%s.pkl' % name)

    print('write time', (dt.now() - t).total_seconds())
