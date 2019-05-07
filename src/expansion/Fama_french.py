import pandas as pd
import pickle
import constants as c


def generate_3_factors(years):
    '''
    Generate all the Fama-French 3-factors financial returns data based on portfolio construction.
    Call this method to generate all the others!
    The first factor is the difference between the market and risk free rate, also called the excess return of
    the market portfolio
    { year : average excess return on market portfolio }
    Second factor is SMB "small minus big". The size of the company is determined by the market cap
    { year : average SMB return value }
    Third factor is HML "high minus low". It is based on the ratio of book equity to market equity value.
    The delineations are as follows:
    High - means that the Book equity / market equity > 70%
    Neutral - 30% <= Book equity / market <= 70%
    Low = book equity / market equity < 30%
    { year : average HML return value }
    :param years: list of years
    :return: Three dictionaries corresponding to excess return, SMB, and HML as values, years as keys
    '''

    print("Generating fama french pickles")
    # Read in Fama French data just once and pass into all methods
    df = pd.read_csv(c.DATA_DIR + 'supplementary_data/fama_french_factors.CSV')
    df.columns = ['year', 'RmRf', 'SMB', 'HML', 'RF']
    df = df.iloc[1:]

    df['year'] = pd.to_datetime(df['year'], format='%Y/%m/%d')
    df['year'] = df['year'].dt.strftime('%Y')

    ffr_factors = df.set_index('year')

    # Calling all methods to generate the Fama-French 3 factors
    generate_RmRf(years, ffr_factors)
    generate_SMB(years, ffr_factors)
    generate_HML(years, ffr_factors)


def generate_RmRf(years, ffr_factors):
    '''
    The first factor is the difference between the market and risk free rate, also called the excess return of
    the market portfolio. Rm is market rate and Rf is risk free rate
    { year : average excess return on market portfolio }
    :param years: list of years,
    :param df: dataframe of all Fama French factors
    :return: Dict with years as keys and values are average excess return on market portfolio
    '''

    # Populate the dictionary with year: Rm-Rf
    RM_RF = {}

    for yr in years:
        # Get Rm-Rf value from column based on the year (rows indexed by years)
        avg_annual_rmrf = ffr_factors.loc[[yr],'RmRf'].tolist()[0]
        RM_RF[yr] = avg_annual_rmrf

    pickle.dump(RM_RF, open(c.RM_RF_PICKLE, 'wb'))

    return RM_RF


def generate_SMB(years, ffr_factors):
    '''
    Second factor is the small minus big value, average return on three small - 3 big portfolios
    Given by the equation SMB = 1/3(small value + small neutral + small growth) - 1/3(big value +
    big neutral + big growth)
    { year : small minus big return value }
    :param years: list of years
    :param df: dataframe with all factors
    :return: Dict with years as keys and values are SMB
    '''

    # Populate the dictionary with year:average SMB value
    SMB = {}

    for yr in years:
        # Again grab the entry in SMB column based on the year (rows indexed by years)
        avg_annual_smb = ffr_factors.loc[[yr],'SMB'].tolist()[0]
        SMB[yr] = avg_annual_smb

    pickle.dump(SMB, open(c.SMB_PICKLE, 'wb'))
    return SMB


def generate_HML(years, ffr_factors):
    '''
    Last factor is HML, high minus low
    Equation is HML = 1/2 (small value + big value) - 1/2(small growth + big growth)
    { year : high minus low value }
    :param years: list of years
    :param df: the data from FamaFrenchFactors.csv
    :return: Dict with years as keys and values are average return on 2 value portfolios minus 2 growth portfolios
    '''

    # Populate the dictionary with year:average annual HML
    HML = {}

    for yr in years:
        # Get entry in the HML based on the year (rows indexed by years)
        avg_annual_hml = ffr_factors.loc[[yr],'HML'].tolist()[0]
        HML[yr] = avg_annual_hml

    pickle.dump(HML, open(c.HML_PICKLE, 'wb'))
    return HML