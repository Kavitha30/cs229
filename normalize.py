import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import constants as c
import src.category_encoding as category_encoding


def run():

	x_train = category_encoding.main(pd.read_csv('train_lendingclub.txt', sep='\t')).drop(c.CATEGORICAL_COLUMNS_ONE_HOT, axis = 1)
	x_val = category_encoding.main(pd.read_csv('validation_lendingclub.txt', sep='\t')).drop(c.CATEGORICAL_COLUMNS_ONE_HOT, axis=1)
	x_test = category_encoding.main(pd.read_csv('test_lendingclub.txt', sep='\t')).drop(c.CATEGORICAL_COLUMNS_ONE_HOT, axis = 1)
	print(x_train.head())




	scaler = StandardScaler()
	x_train[c.NORMALIZE_COLUMNS] = scaler.fit_transform(x_train[c.NORMALIZE_COLUMNS]).round(3)
	# x_train = x_train.drop(c.DROP_COLUMNS, axis=1)
	x_val[c.NORMALIZE_COLUMNS] = scaler.transform(x_val[c.NORMALIZE_COLUMNS]).round(3)
	# x_val = x_val.drop(c.DROP_COLUMNS, axis=1)
	x_test[c.NORMALIZE_COLUMNS] = scaler.transform(x_test[c.NORMALIZE_COLUMNS]).round(3)
	# x_test = x_test.drop(c.DROP_COLUMNS, axis=1)

	x_train.to_csv('train_lendingclub.txt')
	x_test.to_csv('test_lendingclub.txt')
	x_val.to_csv('validation_lendingclub.txt')

	return
if __name__ == '__main__':
    run()
