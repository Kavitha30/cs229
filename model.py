import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.metrics
import sklearn.ensemble
import sys

def get_data():
    X_train = pd.read_pickle('data/x_train.pkl')
    X_val = pd.read_pickle('data/x_val.pkl')
    X_test = pd.read_pickle('data/x_test.pkl')
    Y_train = np.load('data/y_train.npy')
    Y_val = np.load('data/y_val.npy')
    Y_test = np.load('data/y_test.npy')
    datasets = [('Train', X_train, Y_train), ('Val', X_val, Y_val), ('Test', X_test, Y_test)]
    return datasets


def get_model(model_type, param):
    verbosity = 1

    if model_type == 'lr':
        model = sklearn.linear_model.LogisticRegression(penalty='l2', class_weight='balanced', solver='liblinear', 
            n_jobs=-1, max_iter=100, verbose=verbosity, C=1/param)
    elif model_type == 'lr_dumb':
        model = sklearn.linear_model.LogisticRegression(penalty='none', solver='saga', max_iter=100, verbose=verbosity)
    elif model_type == 'rf':
        model = sklearn.ensemble.RandomForestClassifier(class_weight='balanced', n_jobs=6, n_estimators=50, verbose=verbosity,
            max_depth=param)
    elif model_type == 'gb':
        model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=10, max_depth=param, verbose=verbosity)

    return model

def get_metrics(datasets, model, model_type, feature_names):
    aucs = []
    for name, X, Y in datasets:
        pred = model.predict(X)
        auc = sklearn.metrics.roc_auc_score(Y, pred)
        report = sklearn.metrics.classification_report(Y, pred)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y, pred)
        acc = sklearn.metrics.accuracy_score(Y, pred)
        aucs.append(auc)

        print('===================')
        print(name)
        print('===================')
        print('ROC AUC', auc)
        print('ROC fpr', fpr)
        print('ROC tpr', tpr)
        print('ROC thresholds', thresholds)
        print('Accuracy', acc)
        print(report)

    if model_type == 'rf':
        print(list(reversed(sorted(list(zip(model.feature_importances_, feature_names))))))
    return aucs

def main(model_type, params):
    datasets = get_data()
    _, X_train, Y_train = datasets[0]
    print('loaded data')

    best_train, best_val, best_test, best_param = 0, 0, 0, 0
    for param in params:
        print('\n\n===================')
        print(param)

        model = get_model(model_type, param)
        model.fit(X_train, Y_train)
        print('fit model')

        roc_train, roc_val, roc_test = get_metrics(datasets, model, model_type, X_train.columns.values)
        if roc_val > best_val:
            best_train, best_val, best_test, best_param = roc_train, roc_val, roc_test, param

    print('\n\n===================')
    print('Best validation ROC was', best_val, 'with a train ROC of', best_train, 'and test ROC of', best_test, 'for param', best_param)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('You need to specify model type and params!')
    model_type = sys.argv[1]
    params = [float(param) for param in sys.argv[2].split(',')]
    main(model_type, params)
