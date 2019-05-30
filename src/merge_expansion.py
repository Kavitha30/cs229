import constants as c
import pandas as pd
import pickle
import numpy as np


def merge(df):

    print('Unemployment')
    state_to_unemployment = pickle.load(open(c.UNEMPLOYMENT_PICKLE, 'rb'))
    def f1(state, year):
        return state_to_unemployment[state][year]
    df['Unemployment'] = np.vectorize(f1)(df['addr_state'], df['year'])

    print('HPI')
    state_to_hpi = pickle.load(open(c.HPI_PICKLE, 'rb'))
    def f2(state, year):
        return state_to_hpi[state][year]
    df['HPI'] = np.vectorize(f2)(df['addr_state'], df['year'])

    print('SP500')
    year_to_sp500 = pickle.load(open(c.SP500_PICKLE, 'rb'))
    def f3(year):
        return year_to_sp500[str(year)[0:4]]
    df['SP500'] = np.vectorize(f3)(df['year'])

    print('CPI')
    year_to_cpi = pickle.load(open(c.CPI_PICKLE, 'rb'))
    def f4(year):
        return year_to_cpi[str(year)[0:4]]
    df['CPI'] = np.vectorize(f4)(df['year'])

    print('Inflation')
    year_to_inflation = pickle.load(open(c.INFLATION_PICKLE, 'rb'))
    def f5(year):
        return year_to_inflation[str(year)[0:4]]
    df['Inflation'] = np.vectorize(f5)(df['year'])

    print('Federal_fund_rate')
    year_to_ffr = pickle.load(open(c.FFR_PICKLE, 'rb'))
    def f6(year):
        return year_to_ffr[str(year)[0:4]]
    df['Federal_fund_rate'] = np.vectorize(f6)(df['year'])

    print('Corperate_bond_spread')
    year_to_spread = pickle.load(open(c.SPREAD_PICKLE, 'rb'))
    def f7(year):
        return year_to_spread[str(year)[0:4]]
    df['Corperate_bond_spread'] = np.vectorize(f7)(df['year'])

    print('RF_RM')
    year_to_rfrm = pickle.load(open(c.RM_RF_PICKLE, 'rb'))
    def f8(year):
        return float(year_to_rfrm[str(year)[0:4]])
    df['RF_RM'] = np.vectorize(f8)(df['year'])

    print('HML')
    year_to_hml = pickle.load(open(c.HML_PICKLE, 'rb'))
    def f9(year):
        return float(year_to_hml[str(year)[0:4]])
    df['HML'] = np.vectorize(f9)(df['year'])

    print('SMB')
    year_to_smb = pickle.load(open(c.SMB_PICKLE, 'rb'))
    def f10(year):
        return float(year_to_smb[str(year)[0:4]])
    df['SMB'] = np.vectorize(f10)(df['year'])

    print('GDP')
    state_to_gdp = pickle.load(open(c.STATE_GDP, 'rb'))
    def f11(state, year):
        return state_to_gdp[state][year]
    df['GDP'] = np.vectorize(f11)(df['addr_state'], df['year'])

    print('Coincident_economic_activity')
    state_to_cea = pickle.load(open(c.COINCIDENT_EA, 'rb'))
    def f12(state, year):
        return state_to_cea[state][year]
    df['Coincident_economic_activity'] = np.vectorize(f12)(df['addr_state'], df['year'])


    # print('saving')
    # df.to_csv(path_or_buf=c.DATA_DIR + 'complete_dataset_expanded.txt', sep="\t",
    #                encoding='utf-8', index=False)

    return df