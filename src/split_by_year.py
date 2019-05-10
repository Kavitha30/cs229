import numpy as np

def main(dataset):
    '''
    splits given data into one observation per year, labels chargeoff year
    :param dataset: dataset one observation per loan
    :return: df with one observation per loan per active year year
    '''
    dataset = dataset[dataset.last_pymnt_d != 0]
    dataset = dataset.reset_index()
    years = np.array(dataset['last_pymnt_d'] - dataset['issue_d'] + 1)
    dataset['year'] = np.zeros(len(dataset))
    df = dataset.reindex(np.repeat(dataset.index.values, years), method='ffill')
    print('Filling in year values')
    print('on row ...')
    for i in dataset.index.values:
        if i%100000 == 0:
             print(i)
        last_year = dataset.at[i, 'last_pymnt_d']
        first_year = dataset.at[i, 'issue_d']
        df.at[i, 'year'] = np.arange(first_year, last_year + 1)
        for j in range(last_year - first_year):
            df.at[i, 'loan_status'][j] = 'Current'
    df = df.reset_index(drop=True)
    return df