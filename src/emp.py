import numpy as np
import pandas as pd
from sklearn.metrics import auc
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import pickle
import constants as c

ROI_dict = pickle.load(open(c.ROI_DICT, 'rb'))
loss_dict = pickle.load(open(c.LOSS_DICT, 'rb'))
pi_dict = pickle.load(open(c.PI_DICT, 'rb'))

def run():

    subgrades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    optimal_frac = {}

    for grade in subgrades:


        fpr = np.loadtxt('NN_rocs/fpr' + grade + '.npy')
        tpr = np.loadtxt('NN_rocs/tpr' + grade + '.npy')

        print(pi_dict[grade])
        pi_0 = pi_dict[grade][0]
        pi_1 = pi_dict[grade][1]
        ROI = ROI_dict[grade]
        loss_pdf = loss_dict[grade]

        ROC_val = {}
        for l in range(len(loss_pdf)):
            max_t = 0
            ROC_val[l] = 0
            for i in range(len(tpr)):
                t = (l/100)*pi_0 * tpr[i] - ROI * pi_1 * fpr[i]
                if t > max_t:
                    ROC_val[l] = i


        EMP = 0
        for i in range(len(loss_pdf)):
            EMP += ((i/100)*pi_0 * tpr[ROC_val[i]] - ROI * pi_1 * fpr[ROC_val[i]])*loss_pdf[i]

        print('EMP for {}: {}'.format(grade, EMP))

        # fraction of cases
        eta = 0
        for i in range(len(loss_pdf)):
                eta += (pi_0 * tpr[ROC_val[i]] + pi_1 * fpr[ROC_val[i]]) * loss_pdf[i]

        optimal_frac[grade] = eta

        print('optimal fraction for {}: {}'.format(grade, eta))

    fpr = np.load('rf_1_fpr.npy')
    tpr = np.load('rf_1_tpr.npy')

    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)

    i = 0
    while fpr[i] + tpr[i] < eta:
        i += 1

    roc_auc = auc(fpr, tpr)
    roc_label = '{} (AUC={:.3f})'.format('ROC', roc_auc)
    plt.plot(fpr, tpr, color='.75', linewidth=2, label=roc_label)
    colors = {'A':'b', 'B':'g', 'C':'r', 'D':'c', 'E':'m', 'F':'k', 'G':'#f49242'}
    print(optimal_frac)
    for grade in subgrades:
        i = 0
        while fpr[i] + tpr[i] < optimal_frac[grade]:
            i += 1
        plt.plot(fpr[i], tpr[i], marker='o', markersize=6, color=colors[grade], label='Optimal Cutoff for Grade {}'.format(grade))
    plt.legend()
    plt.savefig('roc_cuve_full.png')
    plt.xlim(0,0.12)
    plt.ylim(0, 0.35)
    plt.tight_layout()
    plt.savefig('roc_cuve_zoom.png')

    return

