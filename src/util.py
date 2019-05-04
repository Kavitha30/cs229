import numpy as np
import pandas as pd
import csv
import constants as c

def cat_to_array(df, category):
    '''
    :param df: original dataframe
    :param category: name of categorical variable
    :return: returns a pandas df with an indicator column for each
    category of the given variable.
    '''
    data_rows = df.shape[0]
    indicator_columns = np.zeros((data_rows, 1))
    categories = {df.at[0, category]: 0}
    names = [category + '_' + df.at[0, category]]
    print('on row ...')
    for index, row in df.iterrows():
        if index%50000 == 0:
             print(index)
        if row[category] not in categories:
            names.append(category + '_' + row[category])
            categories[row[category]] = indicator_columns.shape[1]
            indicator_columns = np.concatenate((indicator_columns, np.zeros((data_rows, 1))), axis=1)
        indicator_columns[index, categories[row[category]]] = 1
    return pd.DataFrame(indicator_columns, columns = names)

def binary_encode(df, category):
    data_rows = df.shape[0]
    numbers = [1]
    indicator_columns = np.zeros((data_rows, 1))
    categories = {df.at[0, category]: to_rev_binary(numbers[0])}
    names = [category + '_Bin_0']
    columns = 1
    print('on row ... ')
    for index, row in df.iterrows():
        if index % 50000 == 0:
            print(index)
        if row[category] not in categories:
            if(np.mean(to_rev_binary(numbers[-1])==0) == 0):
                indicator_columns = np.concatenate((indicator_columns, np.zeros((data_rows, 1))), axis=1)
                names.append(category + '_Bin_' + str(columns))
                columns += 1
                for c in categories:
                    categories[c] = np.concatenate((categories[c], np.zeros(1)))
            new_encoding = to_rev_binary(numbers[-1] + 1)
            categories[row[category]] = new_encoding
            numbers.append(numbers[-1] + 1)
        indicator_columns[index] = categories[row[category]]
    return pd.DataFrame(indicator_columns, columns=names)

def to_rev_binary(n):
    s = "{0:b}".format(n)
    l = []
    for i in range(1, len(s)+1):
        l.append(int(s[len(s) - i]))
    return np.array(l)



def load_raw_data():
    print('loading data...')
    '''

    :return: full data set as a pandas df
    '''

    path = c.DATA_DIR
    all_files = []
    for file in c.DATA_SET_FILENAMES:
        all_files.append(path + file)

    li = []
    li.append(pd.read_csv(all_files[0], sep = '\t', index_col=None, header=0))
    li[0].to_pickle(c.DATA_DIR + 'full_data.pkl')
    for filename in all_files[1:]:
        df = pd.read_csv(filename, sep = '\t', index_col=None, names = list(li[0].columns))
        print('reading ' + filename)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True, sort=False)
    frame.to_pickle(c.DATA_DIR + 'full_data.pkl')