import pandas_datareader.data as web
import pickle
import constants as c


def generate(years):
    '''
    Generate list of inflation rates for every year in the list given
    :param years: list of years.
    :return: Dict with mapping year to inflation rate
    Data downloaded from: FRED
    '''
    print('Generating Inflation data')

    start = c.START_DATE
    end = c.END_DATE

    df = web.DataReader("T5YIEM", 'fred', start, end)
    year_to_inf = {}
    for year in years:
        try:
            year_label = str(year) + "-01-01"
            year_to_inf[year] = df.loc[year_label][0]
        except:
            print(str(year) + "-01-01")
            print(df)
    pickle.dump(year_to_inf, open(c.INFLATION_PICKLE, 'wb'))
    return year_to_inf