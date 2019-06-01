import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import constants as c

x_train = pd.read_csv('x_train_exp.csv', sep=',')
x_val = pd.read_csv('x_val_exp.csv', sep=',')
x_test = pd.read_csv('x_test_exp.csv', sep=',')



print('Normalizing')
scaler = StandardScaler()
x_train[c.NORMALIZE_COLUMNS] = scaler.fit_transform(x_train[c.NORMALIZE_COLUMNS]).round(3)
# x_train = x_train.drop(c.DROP_COLUMNS, axis=1)
x_val[c.NORMALIZE_COLUMNS] = scaler.fit_transform(x_val[c.NORMALIZE_COLUMNS]).round(3)
# x_val = x_val.drop(c.DROP_COLUMNS, axis=1)
x_test[c.NORMALIZE_COLUMNS] = scaler.fit_transform(x_test[c.NORMALIZE_COLUMNS]).round(3)
# x_test = x_test.drop(c.DROP_COLUMNS, axis=1)


x_to_write = [
	(x_train, 'x_train_exp.csv'),
	(x_test, 'x_test_exp.csv'),
	(x_val, 'x_val_exp.csv'),
]

print('saving...')
for datasplit, filename in x_to_write:
	datasplit.to_csv(filename, index = False)
print('data saved')