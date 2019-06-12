import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.layers import dense
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from collections import defaultdict

def train(num_epochs, batch_size, X_train, y_train, X_val, y_val, X_test, y_test):
    #initialize model
    lr = tf.placeholder(tf.float32)
    X = tf.placeholder(tf.float32, [None, 89])
    y = tf.placeholder(tf.int32, [None])
    rate = tf.placeholder(tf.float32)

    h1 = tf.layers.dense(inputs=X, units=89, activation=tf.nn.relu)
    #drop1 = tf.nn.dropout(h1, rate)
    h2 = tf.layers.dense(inputs=h1, units=89, activation=tf.nn.relu)
    drop2 = tf.nn.dropout(h2, rate)
    h3 = tf.layers.dense(inputs=drop2, units=45, activation=tf.nn.relu)
    drop3 = tf.nn.dropout(h3, rate)
    h4 = tf.layers.dense(inputs=drop3, units=20, activation=tf.nn.relu)
    #drop4 = tf.nn.dropout(h4, rate)
    pred = tf.layers.dense(inputs=h2, units=2)
    prob = tf.nn.softmax(pred)
    var = tf.trainable_variables() 

    loss = tf.add_n([tf.nn.l2_loss(v) for v in var]) * 0.0001
    loss += tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=tf.one_hot(y, 2), logits=pred, pos_weight=5))

    optimizer = tf.train.AdamOptimizer(lr)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = optimizer.minimize(loss)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init)

    max_step = (X_train.shape[0] + batch_size - 1)//batch_size
    l = []
    a = []
    va = []
    train_auc = []
    val_auc = []
    acc_val = 0

    #train 
    for epoch in range(num_epochs):
        #shuffle before each epoch
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train1 = X_train[indices]
        y_train1 = y_train[indices]

        total_loss, acc = 0, 0
        curr = 0 
        total_pred = np.empty(0)
        for step in range(max_step):
            keys_feed, labels_feed = X_train1[curr:min(curr + batch_size, X_train.shape[0])], y_train1[curr:min(curr + batch_size, X_train.shape[0])]
            curr += batch_size
            feed_dict = {X: keys_feed, y: labels_feed, lr: 0.0001, rate: 0.5}
            _, predictions, curr_loss = sess.run([train_op, pred, loss], feed_dict=feed_dict)
            total_loss += curr_loss
            total_pred = np.append(total_pred, np.argmax(predictions, axis=1), axis=0)
            acc += np.sum(np.argmax(predictions, axis=1) == labels_feed)
            
        print('Epoch %d: loss = %.10f acc = %.10f' % (epoch, total_loss/max_step, acc / X_train.shape[0]))
        l.append(total_loss / max_step)
        a.append(acc / X_train.shape[0])
        train_auc.append(metrics.roc_auc_score(y_train1, total_pred))

        #eval on val
        keys_feed, labels_feed = X_val, y_val
        feed_dict = {X: keys_feed, y: labels_feed, lr: 0, rate: 1}
        probabilities, predictions = sess.run([prob, pred], feed_dict=feed_dict)
        acc = np.sum(np.argmax(predictions, axis=1) == labels_feed)
        print(acc / X_val.shape[0])
        va.append(acc / X_val.shape[0])
        val_auc.append(metrics.roc_auc_score(y_val, np.argmax(predictions, axis=1)))
        if(acc / X_val.shape[0] > acc_val):
            acc_val = acc / X_val.shape[0]
            save_path = saver.save(sess, "/tmp/model.ckpt")

    #plot
    plt.figure()
    plt.plot([i for i in range(num_epochs)], va, label='Validation Accuracy')
    plt.plot([i for i in range(num_epochs)], a, label='Train Accuracy')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.title('Neural Network Accuracy')
    plt.legend()
    plt.savefig('acc.png')

    plt.figure()
    plt.xlabel('Number of Epochs')
    plt.ylabel('Average Loss')
    plt.plot([i for i in range(num_epochs)], l, label='Average Loss')
    plt.title('Neural Network Loss')
    plt.legend()
    plt.savefig('loss.png')

    plt.figure()
    plt.plot([i for i in range(num_epochs)], val_auc, label='Validation AUC')
    plt.plot([i for i in range(num_epochs)], train_auc, label='Train AUC')
    plt.xlabel('Number of Epochs')
    plt.ylabel('AUC')
    plt.title('Neural Network Generalization Gap')
    plt.legend()
    plt.savefig('auc.png')

    saver.restore(sess, "/tmp/model.ckpt")
    #last eval on train
    keys_feed, labels_feed = X_train, y_train
    feed_dict = {X: keys_feed, y: labels_feed, lr: 0, rate: 1}
    probabilities, predictions = sess.run([prob, pred], feed_dict=feed_dict)
    pred_train = np.argmax(predictions, axis=1)

    #last eval on val
    keys_feed, labels_feed = X_val, y_val
    feed_dict = {X: keys_feed, y: labels_feed, lr: 0, rate: 1}
    probabilities, predictions = sess.run([prob, pred], feed_dict=feed_dict)
    pred_val = np.argmax(predictions, axis=1)
    print('val AUC score is {}'.format(metrics.roc_auc_score(y_val, pred_val)))

    #test
    keys_feed, labels_feed = X_test, y_test
    feed_dict = {X: keys_feed, y: labels_feed, lr: 0, rate: 1}
    probabilities, predictions = sess.run([prob, pred], feed_dict=feed_dict)
    pred_test = np.argmax(predictions, axis=1)

    #by-grade to do EMP
    """
    pr = {}
    for g in grade:
        keys_feed, labels_feed = grade[g], y_grade[g]
        feed_dict = {X: keys_feed, y: labels_feed, lr: 0, rate: 1}
        probabilities, predictions = sess.run([prob, pred], feed_dict=feed_dict)
        pr[g] = probabilities[:,1]
        """

    return pred_train, pred_test, probabilities

def results(pred_train, pred_test, y_train, y_test):
    testlen = len(pred_test)
    train_len = len(pred_train)
    accuracy_test = np.sum(pred_test == y_test) / float(testlen)
    accuracy_train = np.sum(pred_train == y_train) / float(train_len)
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

    #Num classfiied to each class
    print(sum([pred_test[i] == 1 for i in range(testlen)]))
    print(sum([pred_test[i] == 0 for i in range(testlen)]))

    print(sum([pred_train[i] == 1 for i in range(train_len)]))
    print(sum([pred_train[i] == 0 for i in range(train_len)]))

    # generate metrics
    print('test accuracy score is {}'.format(metrics.accuracy_score(y_test, pred_test)))
    print('train accuracy score is {}'.format(metrics.accuracy_score(y_train, pred_train)))
    print('test confusion matrix is {}'.format(metrics.confusion_matrix(y_test, pred_test)))
    print('train confusion matrix is {}'.format(metrics.confusion_matrix(y_train, pred_train)))
    print('test AUC score is {}'.format(metrics.roc_auc_score(y_test, pred_test)))
    print('train AUC score is {}'.format(metrics.roc_auc_score(y_train, pred_train)))
    print('report is {}'.format(metrics.classification_report(y_test, pred_test)))

if __name__ == '__main__':
    #preprocessing data
    
    X_train = pd.read_csv('train_lendingclub.txt', sep=',')
    ones = X_train[X_train['loan_status'] == 1]
    ones = ones[ones.columns.difference(['loan_status'])].values
    y_train = X_train['loan_status'].values
    X_train = X_train[X_train.columns.difference(['loan_status'])].values

    X_val = pd.read_csv('validation_lendingclub.txt', sep=',')
    y_val = X_val['loan_status'].values
    X_val = X_val[X_val.columns.difference(['loan_status'])].values

    X_test = pd.read_csv('test_lendingclub.txt', sep=',')
    """
    grade = {'A': [], 'B': [], 'C': [], 'D': [], 'E': [], 'F': [], 'G': []}
    y_grade = {'A': [], 'B': [], 'C': [], 'D': [], 'E': [], 'F': [], 'G': []}
    for g in grade:
        if(g == 'A'):
            grade[g] = X_test[X_test['grade_A'] > 0]
            y_grade[g] = grade[g]['loan_status'].values
        elif(g == 'B'):
            grade[g] = X_test[X_test['grade_B'] > 0]
            y_grade[g] = grade[g]['loan_status'].values
        elif(g == 'C'):
            grade[g] = X_test[X_test['grade_C'] > 0]
            y_grade[g] = grade[g]['loan_status'].values
        elif(g == 'D'):
            grade[g] = X_test[X_test['grade_D'] > 0]
            y_grade[g] = grade[g]['loan_status'].values
        elif(g == 'E'):
            grade[g] = X_test[X_test['grade_E'] > 0]
            y_grade[g] = grade[g]['loan_status'].values
        elif(g == 'F'):
            grade[g] = X_test[X_test['grade_F'] > 0]
            y_grade[g] = grade[g]['loan_status'].values
        elif(g == 'G'):
            grade[g] = X_test[X_test['grade_G'] > 0]
            y_grade[g] = grade[g]['loan_status'].values
        grade[g] = grade[g][grade[g].columns.difference(['loan_status'])].values
    """
    y_test = X_test['loan_status'].values
    X_test = X_test[X_test.columns.difference(['loan_status'])].values

    X_train = np.nan_to_num(X_train)
    ones = np.nan_to_num(ones)
    X_test = np.nan_to_num(X_test)

    m = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    s = np.std(ones, axis=0)
    s[s == 0] = 1
    ones = (ones - np.mean(ones, axis=0)) / s

    X_train = (X_train - m) / std
    X_val = (X_val - m) / std
    X_test = (X_test - m) / std

    """
    for g in grade:
        grade[g] = np.nan_to_num(grade[g])
        grade[g] = (grade[g] - m) / std
    """

    for i in range(3):
        X_train = np.concatenate((X_train, ones), axis=0)
        y_train = np.append(y_train, np.ones(ones.shape[0]))

    pred_train, pred_test, probabilities = train(50, 128, X_train, y_train, X_val, y_val, X_test, y_test)
    results(pred_train, pred_test, y_train, y_test)

    #grade for EMP
    """
    for g in grade:
        fpr, tpr, thresholds = metrics.roc_curve(y_grade[g], probabilities[g], pos_label=1)
        np.savetxt('fpr{}.npy'.format(g), fpr)
        np.savetxt('tpr{}.npy'.format(g), tpr)
        np.savetxt('thresholds{}.npy'.format(g), thresholds)
    """

