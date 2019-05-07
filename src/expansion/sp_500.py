import pandas as pd
import pickle
import constants as c
from datetime import datetime

def generate(years):
    '''
    Generate list of SP500 index values for every day in the list given to us
    :param days: dict of int -> day. Key represents id of row in the data. Value represents day for that loan
    :return: Dict with same keys as days, but with values from SP500.
    Data downloaded from:
    '''
    print("Generating sp500 pickle")
    df = pd.read_csv(c.DATA_DIR + 'supplementary_data/sp-500-historical-annual-returns.csv')
    print(df)
    for _,row in df.iterrows():
        print(row)
        print(row['date'])
        print(row[1])
    annual_return = { str(datetime.strptime(row['date'], "%m/%d/%y").date().year): row[1] for _, row in df.iterrows() }
    year_to_SP500 = {}
    for year in years:
        year_to_SP500[year] = annual_return[year]
    pickle.dump(year_to_SP500, open(c.SP500_PICKLE, 'wb'))
    return year_to_SP500
