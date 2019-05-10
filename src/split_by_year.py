import src.util as util
import constants as c
import pandas as pd
import numpy as np

def main(dataset):
    '''
    splits given data into one observation per year, labels chargeoff year
    :param dataset: dataset one observation per loan
    :return: df with one observation per loan per active year year
    '''
    dataset = dataset[dataset.last_pymnt_d != 0]
    years = np.zeros(len(dataset))
    dataset = dataset.reset_index(drop = True)
    i = 0
    print('generating loan ages')
    print('on row ...')
    for index, row in dataset.iterrows():
        if i%100000 == 0:
             print(i)
        last_year = row['last_pymnt_d']
        first_year = row['issue_d']
        num_rows = last_year - first_year + 1
        years[i] = num_rows
        i+=1
    dataset['years'] = years
    dataset = dataset.reset_index()
    df = dataset.reindex(np.repeat(dataset.index.values, dataset['years']), method='ffill')
    print('Filling in year values')
    print('on row ...')
    for i in dataset.index.values:
        if i%100000 == 0:
             print(i)
        last_year = dataset.at[i, 'last_pymnt_d']
        first_year = dataset.at[i, 'issue_d']
        df.at[i, 'years'] = np.arange(first_year, last_year + 1)
        for j in range(last_year - first_year):
            df.at[i, 'loan_status'][j] = 'Current'
    df = df.reset_index(drop=True)
    return df