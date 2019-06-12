import pandas as pd
import numpy as np
import pickle
import constants as c



def run():
    data = pd.read_csv('train_lendingclub_full.txt', sep='\t')



    ROI = {}
    Loss = {}
    pi = {}
    for grade in data.grade.unique():
        df = data[data['grade'] == grade]

        total = np.sum(df['loan_amnt'])
        print([grade, total])
        def f(A, T):
            ROI =  (T/A) - 1
            return ROI
        df_ROI = df[df['loan_status'] == 0]
        ROI_vec = np.vectorize(f)(df_ROI['loan_amnt'], df_ROI['total_pymnt'])
        ROI[grade] = np.mean(ROI_vec)

        df_loss = df[df['loan_status'] == 1]
        losses =  (1 - (df_loss['total_pymnt']/df_loss['loan_amnt']))
        p0 = np.mean(losses <= 0)
        p1 = np.mean(losses == 1)
        # Loss_uniform[grade] = [p0, p1]
        losses = losses.where(losses >= 0, 0)
        delta = np.arange(0, 1, 0.01)
        delta = np.append(delta, [1.1])
        loss_pdf = np.zeros(100)
        for i in range(len(delta)-1):
            l = losses[(losses>=delta[i]) & (losses<delta[i+1])]
            loss_pdf[i] = l.size/losses.size
        Loss[grade] = loss_pdf

        pi_0 = np.mean(df['loan_status'])
        pi_1 = 1 - pi_0
        pi[grade] = [pi_0, pi_1]


    pickle.dump(ROI, open(c.ROI_DICT, 'wb'))
    pickle.dump(Loss, open(c.LOSS_DICT, 'wb'))
    pickle.dump(pi, open(c.PI_DICT, 'wb'))
    # pickle.dump(Loss_uniform, open(c.LOSS_UNIFORM_DICT, 'wb'))