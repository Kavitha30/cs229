import constants as c
import pandas as pd
import pickle
import numpy as np


def merge(df):

    print('Unemployment')
    state_to_unemployment = pickle.load(open(c.UNEMPLOYMENT_PICKLE, 'rb'))
    def f1(state, issue_d):
        return state_to_unemployment[state][issue_d]
    df['Unemployment'] = np.vectorize(f1)(df['addr_state'], df['issue_d'])

    print('HPI')
    state_to_hpi = pickle.load(open(c.HPI_PICKLE, 'rb'))
    def f2(state, issue_d):
        return state_to_hpi[state][issue_d]
    df['HPI'] = np.vectorize(f2)(df['addr_state'], df['issue_d'])

    print('SP500')
    issue_d_to_sp500 = pickle.load(open(c.SP500_PICKLE, 'rb'))
    def f3(issue_d):
        return issue_d_to_sp500[str(issue_d)[0:4]]
    df['SP500'] = np.vectorize(f3)(df['issue_d'])

    print('CPI')
    issue_d_to_cpi = pickle.load(open(c.CPI_PICKLE, 'rb'))
    def f4(issue_d):
        return issue_d_to_cpi[str(issue_d)[0:4]]
    df['CPI'] = np.vectorize(f4)(df['issue_d'])

    print('Inflation')
    issue_d_to_inflation = pickle.load(open(c.INFLATION_PICKLE, 'rb'))
    def f5(issue_d):
        return issue_d_to_inflation[str(issue_d)[0:4]]
    df['Inflation'] = np.vectorize(f5)(df['issue_d'])

    print('Federal_fund_rate')
    issue_d_to_ffr = pickle.load(open(c.FFR_PICKLE, 'rb'))
    def f6(issue_d):
        return issue_d_to_ffr[str(issue_d)[0:4]]
    df['Federal_fund_rate'] = np.vectorize(f6)(df['issue_d'])

    print('Corperate_bond_spread')
    issue_d_to_spread = pickle.load(open(c.SPREAD_PICKLE, 'rb'))
    def f7(issue_d):
        return issue_d_to_spread[str(issue_d)[0:4]]
    df['Corperate_bond_spread'] = np.vectorize(f7)(df['issue_d'])

    print('RF_RM')
    issue_d_to_rfrm = pickle.load(open(c.RM_RF_PICKLE, 'rb'))
    def f8(issue_d):
        return float(issue_d_to_rfrm[str(issue_d)[0:4]])
    df['RF_RM'] = np.vectorize(f8)(df['issue_d'])

    print('HML')
    issue_d_to_hml = pickle.load(open(c.HML_PICKLE, 'rb'))
    def f9(issue_d):
        return float(issue_d_to_hml[str(issue_d)[0:4]])
    df['HML'] = np.vectorize(f9)(df['issue_d'])

    print('SMB')
    issue_d_to_smb = pickle.load(open(c.SMB_PICKLE, 'rb'))
    def f10(issue_d):
        return float(issue_d_to_smb[str(issue_d)[0:4]])
    df['SMB'] = np.vectorize(f10)(df['issue_d'])

    print('GDP')
    state_to_gdp = pickle.load(open(c.STATE_GDP, 'rb'))
    def f11(state, issue_d):
        return state_to_gdp[state][issue_d]
    df['GDP'] = np.vectorize(f11)(df['addr_state'], df['issue_d'])

    print('Coincident_economic_activity')
    state_to_cea = pickle.load(open(c.COINCIDENT_EA, 'rb'))
    def f12(state, issue_d):
        return state_to_cea[state][issue_d]
    df['Coincident_economic_activity'] = np.vectorize(f12)(df['addr_state'], df['issue_d'])


    # print('saving')
    # df.to_csv(path_or_buf=c.DATA_DIR + 'complete_dataset_expanded.txt', sep="\t",
    #                encoding='utf-8', index=False)

    return df