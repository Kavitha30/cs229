import pickle
import pandas_datareader.data as web
from src.util import process_threaded
import constants as c


def generate(states):
    '''
    :param zip_codes: list of zip codes
    :return: dictionary from zip code to All-Transactions House Price Index for the borrowers
    county in the year of approval. Some adjustments are made for unavailable data (i.e. if HPI is only available back
    to 1993, the 1993 value is used for 1990 - 1993.
    '''
    print('Generating hpi data')
    start = c.START_DATE
    end = c.END_DATE
    args_list = []

    for state in states:
        args_list.append((state, start, end))

    zip_to_hpi = process_threaded(fetch, args_list)
    pickle.dump(zip_to_hpi, open(c.HPI_PICKLE, 'wb'))
    return zip_to_hpi

def fetch(state, start, end):
        '''
        :param states:
        :param start:
        :param end:
        :return:
        '''
        result = {}
        year = int(str(start)[0:4])
        try:
            dataset_name = state + "STHPI"
            data = web.DataReader(dataset_name, 'fred', start, end)
            while year <= int(str(end)[0:4]):
                result[year] = (data.loc[str(year) + "-01-01"][0])
                year = year + 1
        except Exception as e:
            print('Error: State not found: {}'.format(state), e)
            try:
                data = web.DataReader('USSTHPI', 'fred', start, end)
                while year <= int(str(end)[0:4]):
                    result[year] = data.loc[str(year) + "-01-01"][0]
                    year = year + 1
                print('Error corrected: retrieved national data.')
            except Exception as e:
                print('Error: FAILED to get hpi data for state {}. Adding as NA'.format(state), e)
                result = 'NA'

        return state, result