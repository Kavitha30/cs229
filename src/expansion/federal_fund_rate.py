import pandas_datareader.data as web
import pickle
import constants as c


def generate(years):
    '''
    Generate list of immediate fedral fund rates for every year in the list given
    :param years: list of years.
    :return: Dict with mapping year to inflation rate
    Data downloaded from: FRED
    '''
    print('Generating FFR data')

    start = c.START_DATE
    end = c.END_DATE
    # IRLTLT01USA156N, long term federal rate dataset
    df = web.DataReader("IRSTFR01USA156N", 'fred', start, end)
    year_to_ffr = {}
    for year in years:
        year_label = str(year) + "-01-01"
        year_to_ffr[year] = df.loc[year_label][0]
    pickle.dump(year_to_ffr, open(c.FFR_PICKLE, 'wb'))

    return year_to_ffr