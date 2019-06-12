import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import constants as c

data = pd.read_csv(c.DATA_DIR + 'complete_dataset_expanded.txt', sep='\t')
#data = pd.read_csv('lc_test.txt', sep='\t')
#data cleaning
data= data[data.columns.difference(['last_pymnt_amnt'])]
data= data[data.columns.difference(['last_pymnt_d'])]

X = data[data.columns.difference(['loan_status'])]
y = int(data['loan_status'].values)
print(y)

#test/val/train split
print('splitting...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

x_to_write = [
	(X_train, 'x_train_exp.csv'),
	(X_test, 'x_test_exp.csv'),
	(X_val, 'x_val_exp.csv'),
]
y_to_write = [
	(y_train, 'y_train_exp.csv'),
	(y_test, 'y_test_exp.csv'),
	(y_val, 'y_val_exp.csv'),
]

print('saving...')
for datasplit, filename in x_to_write:
	datasplit.to_csv(filename)
for datasplit, filename in y_to_write:
	np.savetxt(filename, datasplit, fmt="%1i")
print('data saved')

# X_train = pd.read_csv('x_train_exp.csv', sep=',')
# print(X_train.head())
# X_test = pd.read_csv('x_test_exp.csv', sep=',')
# y_train = pd.read_csv('y_train_exp.csv', sep=',', header =None)
# y_train = np.ravel(y_train)
# y_test = pd.read_csv('y_test_exp.csv', sep=',', header= None)
# y_test = np.ravel(y_test)

#logistic regression
print('Logistic Regression')
lr = LogisticRegression(solver ='saga')
model = lr.fit(X_train, y_train)
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

# Accuracy
testlen = X_test.shape[0]
train_len =X_train.shape[0]
accuracy_test = sum([pred_test[i] == y_test[i] for i in range(testlen)]) / float(testlen)
accuracy_train = sum([pred_train[i] == y_train[i] for i in range(train_len)]) / float(train_len)
print('Accuracy_test is {}'.format(accuracy_test))
print('Accuracy_train is {}'.format(accuracy_train))

# Specificity: For those who didn't default, how many did it predict correctly?
spec_test = sum([pred_test[i] == y_test[i] and pred_test[i] == 0 for i in range(testlen)]) / float(sum([pred_test[i] == 0 for i in range(testlen)]))
spec_train = sum([pred_train[i] == y_train[i] and pred_train[i] == 0 for i in range(train_len)]) / float(sum([pred_train[i] == 0 for i in range(train_len)]))
print('spec_test is {}'.format(spec_test))
print('spec_train is {}'.format(spec_train))

# Sensitivity: For those who did default, how many did it predict correctly?
sens_test = sum([pred_test[i] == y_test[i] and pred_test[i] == 1 for i in range(testlen)]) / float(sum([pred_test[i] == 1 for i in range(testlen)]))
sens_train = sum([pred_train[i] == y_train[i] and pred_train[i] == 1 for i in range(train_len)]) / float(sum([pred_train[i] == 1 for i in range(train_len)]))
print('sens_test is {}'.format(sens_test))
print('sens_train is {}'.format(sens_train))

# generate metrics
from sklearn import metrics
print('test accuracy score is {}'.format(metrics.accuracy_score(y_test, pred_test)))
print('train accuracy score is {}'.format(metrics.accuracy_score(y_train, pred_train)))
print('test confusion matrix is {}'.format(metrics.confusion_matrix(y_test, pred_test)))
print('train confusion matrix is {}'.format(metrics.confusion_matrix(y_train, pred_train)))
print('test AUC score is {}'.format(metrics.roc_auc_score(y_test, pred_test)))
print('train AUC score is {}'.format(metrics.roc_auc_score(y_train, pred_train)))

#L1 logistic regression
print('L1 Logistic Regression')
from sklearn.linear_model import LogisticRegressionCV
clf1 = LogisticRegressionCV(cv=5, random_state=0, penalty ='l1', solver = 'liblinear').fit(X_train,y_train)
print('R2 for L1 regularization is {}'.format(clf1.score(X, y)))
y_hat_train1 = clf1.predict(X_train)
y_hat_test1 = clf1.predict(X_test)

print('L1 test accuracy score is {}'.format(metrics.accuracy_score(y_test, y_hat_test1)))
print('L1 train accuracy score is {}'.format(metrics.accuracy_score(y_train, y_hat_train1)))
print('L1 test confusion matrix is {}'.format(metrics.confusion_matrix(y_test, y_hat_test1)))
print('L1 train confusion matrix is {}'.format(metrics.confusion_matrix(y_train, y_hat_train1)))
print('L1 test AUC score is {}'.format(metrics.roc_auc_score(y_test, y_hat_test1)))
print('L1 train AUC score is {}'.format(metrics.roc_auc_score(y_train, y_hat_train1)))


#L2 logistic regression
print('L2 Logistic Regression')
clf2 = LogisticRegressionCV(cv=5, random_state=0, penalty ='l2', solver = 'liblinear').fit(X_train,y_train)
print('R2 for L2 regularization is {}'.format(clf2.score(X, y)))
y_hat_train2 = clf2.predict(X_train)
y_hat_test2 = clf2.predict(X_test)

print('L2 test accuracy score is {}'.format(metrics.accuracy_score(y_test, y_hat_test2)))
print('L2 train accuracy score is {}'.format(metrics.accuracy_score(y_train, y_hat_train2)))
print('L2 test confusion matrix is {}'.format(metrics.confusion_matrix(y_test, y_hat_test2)))
print('L2 train confusion matrix is {}'.format(metrics.confusion_matrix(y_train, y_hat_train2)))
print('L2 test AUC score is {}'.format(metrics.roc_auc_score(y_test, y_hat_test2)))
print('L2 train AUC score is {}'.format(metrics.roc_auc_score(y_train, y_hat_train2)))

#gradient boosted trees
print('Gradient Boosted Trees')
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import model_selection
# 10-fold CV, with shuffle
kf_10 = model_selection.KFold( n_splits=10, shuffle=True, random_state=1)

result = []
best_result = 10
best_params = None
best_r_score = None
for l in range(80, 100):
    for k in range(1, 5):
        for l_rate in np.arange(0.1, 1, 0.05):
            regressor = GradientBoostingRegressor(random_state=0, learning_rate = l_rate, n_estimators = l , max_depth = k)
            score = - model_selection.cross_val_score(regressor, X_train, y_train, cv=kf_10, scoring='neg_mean_squared_error').mean()
            r_score = model_selection.cross_val_score(regressor, X_train, y_train, cv=kf_10).mean()
            if score < best_result:
                best_result = score
                best_params = (l, k)
                best_r_score = r_score
            result.append((score, (l, k, l_rate)))
print(best_result)
print(best_params)
print(best_r_score)
