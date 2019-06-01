import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import constants as c

def run():

    X_train = pd.read_csv('x_train_exp.csv', sep=',')
    print(X_train.head())
    X_test = pd.read_csv('x_test_exp.csv', sep=',')
    y_train = pd.read_csv('y_train_exp.csv', sep=',', header =None)
    y_train = np.ravel(y_train)
    y_test = pd.read_csv('y_test_exp.csv', sep=',', header= None)
    y_test = np.ravel(y_test)


    #logistic regression
    print('Logistic Regression')
    lr = LogisticRegression(penalty = 'none', solver ='saga', class_weight='balanced')
    model = lr.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    #print coefficients
    print('Coefficients')
    fieldList = np.array(list(X_train.columns.values)).reshape(-1, 1)
    coeffs = np.reshape(model.coef_, (-1, 1))
    coeffs = np.concatenate((fieldList, coeffs), axis=1)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(pd.DataFrame(coeffs, columns=['Field', 'Coeff']))

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

if __name__ == '__main__':
    run()