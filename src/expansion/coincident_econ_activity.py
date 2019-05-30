import pickle
import pandas_datareader.data as web
from src.util import process_threaded
import constants as c


def generate(states):
    '''
    Generate dict that maps from state to gdp.
    '''
    print('Generating state Coincident Economic Activity')
    start = c.START_DATE.year
    end = c.END_DATE.year
    args_list = []

    for state in states:
        args_list.append((state, start, end))

    state_to_cea = process_threaded(fetch, args_list)

    pickle.dump(state_to_cea, open(c.COINCIDENT_EA, 'wb'))
    return state_to_cea

def fetch(state, start, end) :
    '''
    :param states:
    :param start:
    :param end:
    :return:
    '''
    result = {}
    year = start
    cea_data = state + 'PHCI'
    try:
        cea = web.DataReader(cea_data, 'fred', start, end)
    except:
        cea = web.DataReader('USPHCI', 'fred', start, end)
    while year <= end:
        try:
            result[year] = cea.loc[str(year) + "-01-01"][0]
            year = year + 1
        except:
            result[year] = result[year - 1] + (result[year - 1] - result[year - 2])
            year = year + 1
    return state, result