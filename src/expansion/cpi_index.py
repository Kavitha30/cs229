import pandas as pd
import pickle
import constants as c


def generate(years):
    '''
    Returns average CPI over some year given in the list of dates
    :param dates are list of years
    :return: dictionary with the key:value pair year : average CPI for that year
    '''
    print("Generating Cpi pickle")
    df = pd.read_csv(c.DATA_DIR + 'supplementary_data/cpi_data.csv')

    # Get the year of each reported observation (reported monthly, like 1/31/1990)
    df['FiscalYear'] = pd.to_datetime(df['Date'], format='%Y/%m/%d')
    df['FiscalYear'] = df['FiscalYear'].dt.strftime('%Y')

    # Get the aveage CPI over one year
    cpis = df.groupby(['FiscalYear']).mean()

    CPI_Dict = {}

    for yr in years:                   # For each year, get the value as average CPI of that year
        # The cpis data frame is indexed by year. Get the value in the column 'CPI value'.
        # There was a bug so I had to convert to a list and then take the first entry of that list
        avg_annual_cpi = cpis.loc[[yr],'CPI_value'].tolist()[0]
        #print(avg_annual_cpi)
        CPI_Dict[yr] = avg_annual_cpi
    pickle.dump(CPI_Dict, open(c.CPI_PICKLE, 'wb'))
    return CPI_Dict