import pickle
import pandas_datareader.data as web
from src.util import process_threaded
import constants as c

def generate(states) :
    '''
    Creates a dictionary of all zip codes in existence in 2010 and the county that they are in (in the case that
    a a Zip code area falls into two counties, the dictionary maps to the county where the largest portion falls).
    *Note: Given zip codes from 1990 - 2014, this may be an imperfect mapping as this data is from 2010.
    :return: dictionary mapping zip codes to county FIPS codes;
    '''
    print('Generating unemployment data')
    start = c.START_DATE
    end = c.END_DATE
    args_list = []

    args_list = []

    for state in states:
        args_list.append((state, start, end))

    state_to_unemployment = process_threaded(fetch, args_list)
    pickle.dump(state_to_unemployment, open(c.UNEMPLOYMENT_PICKLE, 'wb'))
    return state_to_unemployment


def fetch(state, start, end) :
    '''
    :param zip: current zip code to create entree for
    :param county:
    :param states:
    :param start:
    :param end:
    :return:
    '''
    result = {}
    year = int(str(start)[0:4])
    try:
        dataset_name = state + "UR"
        data = web.DataReader(dataset_name, 'fred', start, end)
        while year <= int(str(end)[0:4]):
            result[year] = data.loc[str(year) + "-01-01"][0]
            year = year + 1
    except Exception as e:
        print('Error: State not found: {}'.format(state), e)
        try:
            data = web.DataReader('UNRATE', 'fred', start, end)
            while year <= int(str(end)[0:4]):
                result[year] = data.loc[str(year) + "-01-01"][0]
                year = year + 1
            print('Error corrected: retrieved national data.')
        except Exception as e:
            print('Error: FAILED to get unemployment data for state {}. Adding as NA'.format(state), e)
            result = 'NA'

    return state, result