import pickle
import pandas_datareader.data as web
from src.util import process_threaded
import constants as c


def generate(states):
    '''
    Generate dict that maps from state to gdp.
    '''
    print('Generating state GDP data')
    start = c.START_DATE.year
    end = c.END_DATE.year
    args_list = []

    for state in states:
        args_list.append((state, start, end))

    state_to_gdp = process_threaded(fetch, args_list)

    pickle.dump(state_to_gdp, open(c.STATE_GDP, 'wb'))
    return state_to_gdp

def fetch(state, start, end) :
    '''
    :param states:
    :param start:
    :param end:
    :return:
    '''
    result = {}
    year = start
    gdp_data = state + 'NGSP'
    pop_data = state + 'POP'
    gdp = web.DataReader(gdp_data, 'fred', start, end)
    pop = web.DataReader(pop_data, 'fred', start, end)
    while year <= end:
        try:
            result[year] = gdp.loc[str(year) + "-01-01"][0]/pop.loc[str(year) + "-01-01"][0]
            year = year + 1
        except:
            result[year] = result[year - 1] + (result[year - 1] - result[year - 2])
            year = year + 1
    return state, result