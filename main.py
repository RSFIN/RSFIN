# -- coding: utf-8 --

from RSFIN import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import random
def plot_embedding(data, per, label,title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    plt.figure("3D Scatter", facecolor="lightgray")
    ax3d = plt.gca(projection = "3d")
    for i in range(data.shape[0]):
        ax3d.scatter(data[i,0], data[i,1], per[i],
                 color = plt.cm.Set1(label[i] / 10.))
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return ax3d

def Test1(XY, seed = 0, k = 1):
    '''
    Random search rule
    :param XY:  Data set (wiht label)
    :param seed: random seed
    :param k: training\validation set size (in n)
    '''
    X = XY[:, 0:-1]
    Y = XY[:, -1]

    anfis = ANFIS(XY)
    n = X.shape[1]

    expres = 11
    t = 0
    lam = -0.0
    random.seed(seed)
    Val_index, Train_index, Rest_index = anfis.Sample(np.array(list(range(anfis.XY.shape[0]))), int(np.floor(k*n)), int(np.ceil(k*n)))

    while expres > 0 and t <100:
        t += 1
        anfis.MR = anfis.RandomR(6)
        anfis.Train(Train_index, lam = lam)
        res = anfis.Err_Rate(anfis.prediction(X[Val_index]), Y[Val_index], "MRE")
        try:
            if res < expres:
                expres = res
                MR = anfis.MR
        except:
            expres = res
        np.set_printoptions(precision = 3)
    try:
        anfis.MR = MR
    except:
        a = 1
    anfis.Train(np.append(Train_index, Val_index), epoch=30, lam=lam)
    Yp = anfis.prediction(X[Rest_index])

    print(SYSTEM + ", N = " + str(len(Val_index) + len(Train_index)) + ", MRE = " + str(
        round(anfis.Err_Rate(Yp, Y[Rest_index], "MRE"), 3)) + ", R2 = " + str(
        round(anfis.Err_Rate(Yp, Y[Rest_index], "R2"), 3)))

    return round(anfis.Err_Rate(Yp, Y[Rest_index], "R2"), 3)

    plt.plot(np.array(list(range(len(Yp)))) + 1, Y[Rest_index])
    plt.plot(np.array(list(range(len(Yp))))+1, Yp)
    plt.legend(["real","prediction"])
    plt.title(SYSTEM + ", N = " + str(len(Val_index) + len(Train_index)) + ", MRE = " + str(
        round(anfis.Err_Rate(Yp, Y[Rest_index], "MRE"), 3)) + ", $R^2$ = " + str(
        round(anfis.Err_Rate(Yp, Y[Rest_index], "R2"), 3)))
    plt.show()
def Test2(XY, seed = 0, k = 1, epcho = 500):
    '''
    Search rule by entropy
    :param XY:  Data set (wiht label)
    :param seed: random seed
    :param k: training\validation set size (in n)
    '''

    XY[:, -1] =  XY[:, -1] + 0.1*max(XY[:, -1]) # Reduce the zero sensitivity of MRE
    X = XY[:, 0:-1]
    Y = XY[:, -1]

    anfis = ANFIS(XY)
    n = X.shape[1]

    lam = -0.00
    random.seed(seed)

    Val_index, Train_index, Rest_index = anfis.Sample(np.array(list(range(anfis.XY.shape[0]))), int(np.floor(k*n)), int(np.ceil(k*n)))

    # Constructing rule layer based on training samples
    print('Constructing rules...\n')
    R_index = np.append(Train_index,Val_index)
    if k > 3:
        epcho = 100
    MR_b, Err = [], np.inf
    for i in trange(epcho):
        random.shuffle(R_index)
        anfis.MR = anfis.X[R_index[:2]]
        anfis.Train(Train_index, epoch=1, lam=lam)
        Err_t = anfis.Err_Rate(anfis.prediction(X[Val_index]), Y[Val_index], "All")

        if Err_t < Err:
            MR_b, Err = anfis.MR, Err_t
        if Err < 1:
            print('Complete rule construction in advance\n')
            break
    anfis.MR = MR_b

    # Training RSFIN
    print('\n Training model...')
    anfis.Train(np.append(Train_index, Val_index), epoch=60, lam = lam)
    Yp = anfis.prediction(X[Rest_index])

    print(SYSTEM + ", N = " + str(len(Val_index) + len(Train_index)) + ", MRE = " + str(
        round(anfis.Err_Rate(Yp, Y[Rest_index], "MRE"), 3)) + ", R^2 = " + str(
        round(anfis.Err_Rate(Yp, Y[Rest_index], "R2"), 3)))

    # return anfis.Err_Rate(Yp, Y[Rest_index], "MRE")

    # Drawing comparison diagram
    plt.plot(np.array(list(range(len(Yp)))) + 1, Y[Rest_index])
    plt.plot(np.array(list(range(len(Yp)))) + 1, Yp)
    plt.legend(["real","prediction"])
    plt.title(SYSTEM + ", N = " + str(len(Val_index) + len(Train_index)) + ", MRE = " + str(
        round(anfis.Err_Rate(Yp, Y[Rest_index], "MRE"), 3)) + ", $R^2$ = " + str(
        round(anfis.Err_Rate(Yp, Y[Rest_index], "R2"), 3)))
    plt.xlabel('Configuration index')
    plt.ylabel('Performance')
    plt.show()

if __name__ == '__main__':

    SYSTEM = 'Apache'
    k = 2.5
    PATH = 'data/' + SYSTEM + '_AllNumeric.csv'

    df = pd.read_csv(PATH)
    XY = np.array(df)
    Test2(XY, 0, k)


