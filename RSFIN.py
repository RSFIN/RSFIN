# -- coding: utf-8 --
'''
@File       :   RSFIN.py
@Author     :   Lyle
@Modify Time:   2022/7/14 16:27
'''

import numpy as np
import random
from time import time
import scipy.stats as st
from scipy.optimize import leastsq
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error as mape
class ANFIS:

    def __init__(self, XY, *MR, seed=int(time())):
        np.random.seed(seed)
        random.seed(seed)
        if len(MR) > 0:
            self.MR = np.array(MR[0])
            if len(np.shape(self.MR)) == 1:
                self.MR = np.reshape(self.MR, (1, len(self.MR)))
        else:
            self.MR = np.array([])

        self.XY = np.array(XY)
        self.Setup()

    def Setup(self):
        self.X = self.XY[:, 0:-1]
        self.Y = self.XY[:, -1]

        Invalid_Data_Index = []
        for i in range(self.X.shape[1]):
            if len(np.unique(self.X[:, i])) == 1:
                Invalid_Data_Index.append(i)
        self.X = np.delete(self.X, Invalid_Data_Index, axis=1)
        self.Invalid_Data_Index = Invalid_Data_Index

        self.Xmax = np.max(self.X, axis=0)
        self.Xmin = np.min(self.X, axis=0)
        self.X = (self.X - self.Xmin) / (self.Xmax - self.Xmin)

        self.Ymax = np.max(self.Y)
        self.Ymin = np.min(self.Y)
        # self.Y = (self.Y - self.Ymin) / (self.Ymax - self.Ymin)
        self.Y = self.Y / self.Ymax

        self.MeanShift()
        self.X2mf()
    def Train(self, Train_index, epoch = 5, lam = 0.0):

        vdw = sdw = 0
        for t in range(1, epoch + 1):
            self.mFun(self.X[Train_index])
            bMR = self.reMR()

            O_1 = np.copy(self.F)
            O_2 = np.exp(np.dot(bMR.transpose(), np.log(O_1 + 0.0001)))
            O_3 = O_2 / np.dot(np.ones([O_2.shape[0], O_2.shape[0]]), O_2)

            self.Structure_after(self.X[Train_index], self.Y[Train_index], O_3, lam)


            O_4 = np.multiply(np.dot(self.Ac.transpose(), np.append(self.X[Train_index], np.ones(
                [self.X[Train_index].shape[0], 1]), axis=1).transpose()), O_3)

            O_5 = np.sum(O_4, axis=0)

            if self.Err_Rate(O_5, self.Y[Train_index]) < 0.001:
                break

            vdw, sdw = self.reMF(Train_index, bMR, O_5, vdw, sdw, t)
    def Perceive(self, Test_X):

        if len(np.shape(Test_X)) == 1:
            Test_X = np.reshape(Test_X, (1, len(Test_X)))
        Test_X = (Test_X - self.Xmin) / (self.Xmax - self.Xmin)

        self.mFun(Test_X)
        bMR = self.reMR()

        O_1 = np.copy(self.F)
        O_2 = np.exp(np.dot(bMR.transpose(), np.log(O_1 + 0.0001)))
        O_3 = O_2 / np.dot(np.ones([O_2.shape[0], O_2.shape[0]]), O_2)
        # O_4 = np.multiply(
        #     np.dot(self.Ac.transpose(), np.append(Test_X, np.ones([Test_X.shape[0], 1]), axis=1).transpose())
        #     , O_3)

        return O_3
    def prediction(self, Test_X):

        if len(np.shape(Test_X)) == 1:
            Test_X = np.reshape(Test_X, (1, len(Test_X)))
        if Test_X.shape[1] > self.X.shape[1]:
            Test_X = np.delete(Test_X, self.Invalid_Data_Index, axis = 1)
        Test_X = (Test_X - self.Xmin) / (self.Xmax - self.Xmin)

        self.mFun(Test_X)
        bMR = self.reMR()

        O_1 = np.copy(self.F)
        O_2 = np.exp(np.dot(bMR.transpose(), np.log(O_1 + 0.0001)))
        O_3 = O_2 / np.dot(np.ones([O_2.shape[0], O_2.shape[0]]), O_2)
        O_4 = np.multiply(
            np.dot(self.Ac.transpose(), np.append(Test_X, np.ones([Test_X.shape[0], 1]), axis=1).transpose())
            , O_3)
        O_5 = np.sum(O_4, axis=0)

        # return O_5 * (self.Ymax - self.Ymin) + self.Ymin
        return O_5 * self.Ymax
    def x2MR(self, x):
        MR = np.zeros(x.shape)
        for i in range(len(x)):
            m = np.zeros(len(self.mf[i].mf))
            for j in range(len(m)):
                m[j] = self.MF(x[i], self.mf[i].mf[j].type, self.mf[i].mf[j].config)
            MR[i] = np.argmax(m)
        return MR
    def reMF(self, Train_index, bMR, Predicted_Y, vdw, sdw, t):
        s, beta1, beta2, alpha, epsilon = 0, 0.9, 0.9, 0.001, 0.001
        error = Predicted_Y - self.Y[Train_index]
        F = self.F + 0.001

        D_1 = np.dot(bMR, np.multiply(np.dot(self.Ac.transpose(),
                                             np.append(self.X[Train_index], np.ones([len(Train_index), 1]),
                                                       axis=1).transpose()),
                                      np.exp(np.dot(bMR.transpose(), np.log(F))))) / F
        D_2 = np.dot(1 - bMR, np.multiply(
            np.dot(self.Ac.transpose(),
                   np.append(self.X[Train_index], np.ones([len(Train_index), 1]), axis=1).transpose()),
            np.exp(np.dot(bMR.transpose(), np.log(F)))))
        D_3 = np.dot(bMR, np.exp(np.dot(bMR.transpose(), np.log(F)))) / F
        D_4 = np.dot(1 - bMR, np.exp(np.dot(bMR.transpose(), np.log(F))))

        q = np.multiply(D_2, D_3) / (D_1 + 0.001) - D_4
        a = D_3
        b = D_4
        dYdF = np.multiply(q, a) / (np.multiply(q, F) + b + 0.001) ** 2

        dc = []
        dsig = []

        for i in range(len(self.mf)):
            x = self.X[Train_index, i]
            for j in range(len(self.mf[i].mf)):
                sig = self.mf[i].mf[j].config[0]
                c = self.mf[i].mf[j].config[1]
                dFdsig = np.multiply(np.exp(-(x - c) ** 2 / 2 / sig ** 2), (x - c) ** 2) / sig ** 3
                dFdc = np.multiply(np.exp(-(x - c) ** 2 / 2 / sig ** 2), (x - c)) / sig ** 2
                dsig.append(np.dot(error, np.multiply(dYdF[s, :], dFdsig).transpose()))
                dc.append(np.dot(error, np.multiply(dYdF[s, :], dFdc).transpose()))
                s += 1
        dw = np.append(dsig, dc)
        vdw = beta1 * vdw + (1 - beta1) * dw
        sdw = beta2 * sdw + (1 - beta2) * dw ** 2
        vdwc = vdw / (1 - beta1 ** t)
        sdwc = sdw / (1 - beta2 ** t)

        delta = alpha * vdwc / (np.sqrt(sdwc) + epsilon)
        dsig = delta[:len(dc)]
        dc = delta[len(dc):]

        s = 0
        for i in range(len(self.mf)):
            for j in range(len(self.mf[i].mf)):
                self.mf[i].mf[j].config[0] -= dsig[s]
                self.mf[i].mf[j].config[1] -= dc[s]
                s += 1
        return vdw, sdw
    def Structure_after(self, X, Y, O_3, lam):
        x = y = []
        for k in range(X.shape[0]):
            x_t = []
            for i in range(O_3.shape[0]):
                x_t = np.append(x_t, np.append(X[k], 1) * O_3[i, k])
            x_t = np.array(x_t).reshape(1, len(x_t))
            if len(x) == 0:
                x = x_t
            else:
                x = np.append(x, x_t, axis=0)
            y = np.append(y, Y[k])

        y = y.reshape(len(y), 1)
        self.Ac = np.dot(np.dot(np.linalg.pinv(np.dot(x.transpose(), x) - lam * np.eye(x.shape[1])), x.transpose()), y).reshape(
            O_3.shape[0], X.shape[1] + 1).transpose()
    def Train4MR(self, Train_index, Val_index, epoch, lam = 0):
        self.Train(Train_index, lam = lam)
        for t in range(epoch):
            if len(self.MR) == 0:
                nR = [self.GenMRbyEntropy(Train_index, Val_index, lam = lam)]
                nMR = self.Update_MR(np.array(nR))
            else:
                nR = self.AddMRbyR(Train_index, Val_index, lam = lam)
                nMR = self.Update_MR(np.append(self.MR, [nR], axis=0))
            self.MR = nMR
            self.Train(Train_index, lam = lam)
            if len(self.MR) >= 2:
                self.DelMRbyT(Train_index, Val_index, lam=lam)
            self.Train(Train_index, lam=lam)
            # print(round(self.Err_Rate(self.prediction(self.X[Val_index]), self.XY[Val_index, -1]),3))

        return False
    def GenMRbyR(self, Train_index, Val_index, lam = 0):
        score_o = np.inf
        for i in range(20):
            RMR = self.RandomR(2)
            self.MR = RMR
            self.Train(Train_index, lam=lam)
            score = round(self.Err_Rate(self.prediction(self.X[Val_index]), self.XY[Val_index, -1]), 3)
            if score < score_o:
                score_o = score
                bMR = RMR
        return bMR
    def GenMRbyEntropy(self, Train_index, Val_index, lam = 0):
        epsilion = 0.33
        delta = np.std(self.Y[Train_index])/3
        g = lambda c, mean, cov: st.multivariate_normal.pdf(c, mean, cov) * (2 * np.pi) ** (len(c)/2) * (np.linalg.det(cov) ** (1/2))
        h = lambda c, mean, sigma: st.norm.pdf(c, mean, sigma) * (2 * np.pi) ** (1/2) * sigma

        # R_index = np.append(Train_index, Val_index)
        R_index = Train_index

        random.shuffle(np.array(R_index))
        CradleMR = self.X[R_index[:]]
        # CradleMR = self.RandomR(20)

        for R in CradleMR:
            H = 0
            center = np.array([self.CluRe.C[i][int(R[i])] for i in range(len(R))])
            v_u, v_d = 0, 0
            for index in Train_index:
                a = g(self.X[index], center, epsilion * np.eye(len(R)))
                for oR in self.MR:
                    a *= (1-g(self.X[index],self.r2x(oR),epsilion * np.eye(len(R))))
                b = h(self.Y[index], (self.prediction(center) - self.Ymin) / (self.Ymax - self.Ymin), delta)
                v_u += a*b
                v_d += a
            p = v_u/v_d
            H = -p*np.log(p)

            try:
                S = np.append(S, H)
            except:
                S = H

        # print(S)
        nR = CradleMR[np.argmin(S)]
        while any(np.array_equal(nR, R) for R in self.MR):
            S[np.armax(S)] = -np.inf
            if np.max(S) < 0:
                print("An error occurred while generating the rule!")
                return False
            else:
                nR = CradleMR[np.argmax(S)]
        return nR
    def AddMRbyR(self, Train_index, Val_index, lam = 0):
        CradleMR = self.RandomR(20)
        tempMR = self.MR
        for R in CradleMR:
            self.MR = self.Update_MR(np.append(self.MR, [R], axis=0))
            self.Train(Train_index, lam = lam)
            score = round(self.Err_Rate(self.prediction(self.X[Val_index]), self.XY[Val_index, -1], "MRE"), 3)
            self.MR = tempMR
            self.Train(Train_index, lam = lam)
            try:
                Q = np.append(Q, score)
            except:
                Q = score
        nR = CradleMR[np.argmin(Q)]
        while any(np.array_equal(nR, R) for R in self.MR):
            Q[np.argmin(Q)] = np.inf
            if np.min(Q) >= 1e17:
                print("An error occurred while generating the rule!")
                return False
            else:
                nR = CradleMR[np.argmin(Q)]
        return nR
    def DelMRbyT(self, Train_index, Val_index, lam = 0):
        tempMR = self.MR
        score_o = round(self.Err_Rate(self.prediction(self.X[Val_index]), self.XY[Val_index, -1]), 3)
        flag = 0
        for i in range(len(self.MR)):
            self.MR = np.delete(self.MR, i, axis = 0)
            self.Train(Train_index, lam=lam)
            score = round(self.Err_Rate(self.prediction(self.X[Val_index]), self.XY[Val_index, -1]), 3)
            self.MR = tempMR
            self.Train(Train_index, lam=lam)
            if score < score_o:
                flag = 1
                score_o = score
                del_pos = i

        if flag == 1:
            self.MR = np.delete(self.MR, del_pos, axis=0)
    def RandomR(self, size):
        MR = []
        for t in range(size):
            x = np.array([random.uniform(0,1) for _ in range(self.X.shape[1])])
            try:
                MR = np.append(MR, [self.x2MR(x)], axis=0)
            except:
                MR = np.reshape(self.x2MR(x),(1, len(self.x2MR(x))))

        return self.Union(MR)
    def r2x(self, r):
        x = []
        for i in range(len(r)):
            xi = self.mf[i].mf[int(r[i])].config[1]
            xi = xi*(self.Xmax[i] - self.Xmin[i]) +  self.Xmin[i]
            x.append(xi)
        return np.array(x)
    def Update_MR(self, Updated_MR):
        socer = self.Rules_Socer(self.X[np.random.choice(list(range(self.X.shape[0])), size=100, replace=True)],
                                 Updated_MR)
        # Effective_rules = np.array(np.where(socer > 0.9))
        # if len(Effective_rules[0]) > 0 and len(Updated_MR) > 1:
            # Updated_MR = np.delete(Updated_MR, np.argmin(socer), axis=0)
        return self.Union(Updated_MR)
    def Rules_Socer(self, X_sample, MR):
        O_2, O_3 = self.First_Three_Layers(X_sample, MR)
        socer = np.max(O_3, axis = 1)
        return socer
    def First_Three_Layers(self, X_sample, MR):
        self.mFun(X_sample)
        bMR = self.reMR(MR)
        O_1 = np.copy(self.F)
        O_2 = np.exp(np.dot(bMR.transpose(), np.log(O_1 + 0.0001)))
        O_3 = O_2 / np.dot(np.ones([O_2.shape[0], O_2.shape[0]]), O_2)
        return O_2, O_3
    def mFun(self, X):

        F = Ft = Aux = []
        for x_n in range(X.shape[0]):
            Aux = []
            Ft = []
            for i in range(len(self.mf)):
                Aux.append(len(Ft) + 1)
                for j in range(len(self.mf[i].mf)):
                    Ft.append(self.MF(X[x_n, i],
                                      self.mf[i].mf[j].type,
                                      self.mf[i].mf[j].config))
            F.append(Ft)
        self.F = np.reshape(F, [X.shape[0], len(Ft)]).transpose()
        self.Aux = Aux
    def reMR(self, *MR):
        if len(MR) == 0:
            MR = self.MR
        else:
            MR = MR[0]

        bMR = np.zeros([self.F.shape[0], MR.shape[0]])
        for i in range(MR.shape[0]):
            for j in range(MR.shape[1]):
                bMR[int(np.min([self.F.shape[0], self.Aux[j] + MR[i, j]])) - 1, i] = 1

        return bMR
    def MeanShift(self):
        class CLURE:
            def __init__(self):
                self.C = []

            def append(self, C):
                self.C.append(C)

        self.CluRe = CLURE()

        for i in range(self.X.shape[1]):
            CluRe = self.MS(self.X[:, i])
            if len(CluRe) == 0:
                CluRe = np.array([np.min(self.X[0, i]), np.max(self.X[0, i]) + 0.1])
            self.CluRe.append(CluRe)
    def X2mf(self):

        class MUFUN:
            def __init__(self, type, config):
                self.type = type
                self.config = config

        class MF:
            def __init__(self):
                self.mf = []

            def append(self, mf):
                self.mf.append(mf)

        CC = []
        for i in range(self.X.shape[1]):
            C = np.cov([self.X[:, i], self.Y])
            CC.append(C[0, 1])

        self.mf = []
        for i in range(self.X.shape[1]):
            self.mf.append(MF())
            for j in range(len(self.CluRe.C[i])):
                type = 'gaussmf'
                config = [np.sqrt((self.CluRe.C[i][1] - self.CluRe.C[i][0])) / np.max([np.abs(CC[i]),1]),
                          self.CluRe.C[i][j]]
                self.mf[i].append(MUFUN(type, config))
    def Err_Rate(self, Predicted_Y, Real_Y, outtype = "All"):

        R2 = (1 - r2_score(Real_Y, Predicted_Y)) * 100
        MRE = np.mean(np.divide(np.abs(Real_Y-Predicted_Y), Real_Y+0.001)) * 100

        if outtype == "R2":
            return R2
        elif outtype == "MRE":
            return MRE
        elif outtype == "All":
            return min(R2, MRE)
        else:
            print("An error occurred while calculating the error!")
            return False

    def __str__(self):
        MR = np.array([self.r2x(r) for r in self.MR])
        Ac = self.Ac.transpose()
        res = []
        for r in MR:
            res.append(self.prediction(r))

        df = pd.DataFrame()  # 或df = pd.DataFrame(columns=('A','B'))
        for i in range(len(MR[0])):
            z = dict()
            z[' '] = ['option ' + str(i + 1)]
            for j in range(MR.shape[0]):
                t = round(Ac[j, i], 2)
                if t < 0:
                    space = ''
                else:
                    space = ' '
                z['rule' + str(j + 1)] = [str(int(MR[j, i])) + ' (' + space + str(t) + ')']

            df = df.append(pd.DataFrame(z), ignore_index=True)
        z = dict()
        z[' '] = ['PERF']
        for i in range(MR.shape[0]):
            z['rule' + str(i + 1)] = [round(float(res[i]), 2)]

        df = df.append(pd.DataFrame(z), ignore_index=True)
        return str(df)
    @classmethod
    def infocls(cls):
        print(cls)
    @staticmethod
    def MS(x):
        m = 1
        if np.exp(np.max(x)) >= 1e5:
            m = np.max(x)
            x = x / m

        min_x = np.min(x)
        max_x = np.max(x)

        r = 1 / 9 * (max_x - min_x)
        CluRe = np.linspace(min_x, max_x, 10)
        pCluRe = CluRe

        while 1:
            dis = np.abs(np.log(np.dot(np.exp(-np.matrix(x).transpose()), np.exp(np.matrix(CluRe)))))
            ab = np.array(np.where(dis < r))
            a = ab[0, :]
            b = ab[1, :]
            label = np.unique(b)

            for i in label:
                CluRe[i] = np.mean(x[a[np.where(b == i)]])

            if np.max(np.abs(CluRe - pCluRe)) < 1e-10:
                CluRe = np.unique(CluRe[label]) * m
                return CluRe
            CluRe = np.unique(CluRe[label])
            pCluRe = CluRe
    @staticmethod
    def MF(x, type, config, *fun):
        if len(fun) == 0:
            fun = 1

        if type == 'gaussmf':
            if fun == 0:
                return ['exp(-(x-c).^2/2/sig^2', {'c', 'sig'}]
            sig = config[0]
            c = config[1]
            return np.exp(-(x - c) ** 2 / 2 / sig ** 2)
    @staticmethod
    def Union(MR):
        Check_array = np.dot(np.exp(-MR), np.exp(MR.transpose())) - MR.shape[1]
        ij = np.array(np.where(Check_array == 0))
        i, j = ij[0, :], ij[1, :]
        j = np.delete(j, np.where(i - j >= 0), axis=0)
        return np.delete(MR, j, axis = 0)
    @staticmethod
    def Sample(Available_index_set, Val_size, Train_size):

        if len(Available_index_set) < Val_size + Train_size:
            print('Check the data set size!')

        Available_index_set = np.array(Available_index_set)
        Val_index_pos = random.sample(list(range(len(Available_index_set))), Val_size)
        Val_index = Available_index_set[Val_index_pos]
        Available_index_set = np.delete(Available_index_set, Val_index_pos)

        Train_index_pos = random.sample(list(range(len(Available_index_set))), Train_size)
        Train_index = Available_index_set[Train_index_pos]
        Rest_index = np.delete(Available_index_set, Train_index_pos)

        return Val_index, Train_index, Rest_index

if __name__ == '__main__':

    XY = np.array([[0, 0, 1],
                   [0, 1, 0]])



