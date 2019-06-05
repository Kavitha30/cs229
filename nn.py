import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.layers import dense
import pandas as pd
from sklearn import metrics

max_step = 60
X_train = pd.read_csv('x_train_exp.csv', sep=',')
X_train = X_train[X_train.columns.difference(['out_prncp'])].values
X_val = pd.read_csv('x_val_exp.csv', sep=',')
X_val = X_val[X_val.columns.difference(['out_prncp'])].values
X_test = pd.read_csv('x_test_exp.csv', sep=',')
X_test = X_test[X_test.columns.difference(['out_prncp'])].values
y_train = pd.read_csv('y_train_exp.csv', sep=',', header =None)
y_train = np.ravel(y_train)
y_val = pd.read_csv('y_val_exp.csv', sep=',', header =None)
y_val = np.ravel(y_val)
y_test = pd.read_csv('y_test_exp.csv', sep=',', header= None)
y_test = np.ravel(y_test)

def train(num_epochs, batch_size):
    #initialize model
    lr = tf.placeholder(tf.float32)
    X = tf.placeholder(tf.float32, [None, 150])
    y = tf.placeholder(tf.int32, [None])

    h1 = tf.layers.dense(inputs=X, units=150, activation=tf.nn.relu)
    h2 = tf.layers.dense(inputs=h1, units=150, activation=tf.nn.relu)
    h3 = tf.layers.dense(inputs=h2, units=75, activation=tf.nn.relu)
    h4 = tf.layers.dense(inputs=h3, units=25, activation=tf.nn.relu)
    pred = tf.layers.dense(inputs=h4, units=2)
    prob = tf.nn.softmax(pred)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred))

    optimizer = tf.train.AdamOptimizer(lr)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = optimizer.minimize(loss)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    #train 
    curr = 0
    for epoch in range(num_epochs):
        total_loss, acc = 0, 0
        for step in range(max_step):
            keys_feed, labels_feed = X_train[curr:curr + batch_size], y_train[curr:curr + batch_size]
            curr += batch_size
            feed_dict = {X: keys_feed, y: labels_feed, lr: 0.01}
            _, predictions, curr_loss = sess.run([train_op, pred, loss], feed_dict=feed_dict)
            total_loss += curr_loss
            acc += np.sum(np.argmax(predictions, axis=1) == labels_feed)
            
        print('Epoch %d: loss = %.10f acc = %.10f' % (epoch, total_loss/max_step, acc/(max_step * batch_size)))
    
    keys_feed, labels_feed = X_train[curr:], y_train[curr:]
    feed_dict = {X: keys_feed, y: labels_feed, lr: 0.01}
    _, predictions, curr_loss = sess.run([train_op, pred, loss], feed_dict=feed_dict)

    #eval on train
    keys_feed, labels_feed = X_train, y_train
    feed_dict = {X: keys_feed, y: labels_feed, lr: 0.01}
    probabilities, predictions = sess.run([prob, pred], feed_dict=feed_dict)
    pred_train = np.argmax(predictions, axis=1)

    #test
    keys_feed, labels_feed = X_test, y_test
    feed_dict = {X: keys_feed, y: labels_feed, lr: 0.01}
    probabilities, predictions = sess.run([prob, pred], feed_dict=feed_dict)
    pred_test = np.argmax(predictions, axis=1)

    return pred_train, pred_test, probabilities

def results(pred_train, pred_test):
    testlen = X_test.shape[0]
    train_len =X_train.shape[0]
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

    # generate metrics
    print('test accuracy score is {}'.format(metrics.accuracy_score(y_test, pred_test)))
    print('train accuracy score is {}'.format(metrics.accuracy_score(y_train, pred_train)))
    print('test confusion matrix is {}'.format(metrics.confusion_matrix(y_test, pred_test)))
    print('train confusion matrix is {}'.format(metrics.confusion_matrix(y_train, pred_train)))
    print('test AUC score is {}'.format(metrics.roc_auc_score(y_test, pred_test)))
    print('train AUC score is {}'.format(metrics.roc_auc_score(y_train, pred_train)))

if __name__ == '__main__':
    pred_train, pred_test, probabilities = train(100, 131)
    results(pred_train, pred_test)

