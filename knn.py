import numpy as np
import pandas as pd
import time
from scipy.spatial.distance import euclidean
from scipy import stats
from matplotlib import pyplot as plt

def calc_all_distancies(data_x, unknown):
    num_data = data_x.shape[0]
    num_pred = unknown.shape[0]
    dists = np.zeros((num_pred, num_data))
    for i in range(num_pred):
        for j in range(num_data):
            dists[i,j] = euclidean(unknown[i], data_x[j])
    return dists

def predict(dists, data_y, k):
    num_pred = dists.shape[0]
    y_pred = np.zeros(num_pred)
    for j in range(num_pred):
        dst = dists[j]
        y_closest = data_y[np.argsort(dst)[:k]]
        y_pred[j] = stats.mode(y_closest).mode
    return y_pred

def accuracy(predicted,real):
    total = len(real)
    s = sum(real == predicted)
    return 100*s/total


def compare_k(data_x, data_y, test_x, test_y, kmin=1, kmax=50, kstep=4):
    k = list(range(kmin, kmax, kstep))
    steps = len(k)
    features = np.zeros((steps,3))
    print('Evaluating distancies started')
    
    t0 = time.time()
    distancies = calc_all_distancies(data_x,test_x)
    miss = []
    t = time.time()
    s1 = data_x.shape[0]
    s2 = test_x.shape[0]
    print('Distancies completed in %d seconds for %dx%d' %(t-t0,s1,s2))
    
    for j in range(steps):
        t0 = time.time()
        yk = predict(distancies,data_y,k[j])
        t = time.time() - t0
        features[j][0] = k[j]
        features[j][1] = accuracy(yk,test_y)
        features[j][2] = t
        cond = yk!=test_y
        miss.append({
            'k':k[j],
            'acc':features[j][1],
            'x':test_x[cond]}
        )
        
        print('k={0}, accuracy = {1}%, time = {2} sec'.format(k[j],features[j][1],features[j][2]))
    return features, miss

num_observations = 300
x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
x2 = np.random.multivariate_normal([-2, 3], [[2, .75], [.75, 2]], num_observations)

X = np.vstack((x1, x2)).astype(np.float32)
Y = np.hstack((np.zeros(num_observations),
               np.ones(num_observations)))

ratio = 0.67
n_trn = int(ratio*num_observations)
ind = np.random.permutation(num_observations)

X= X[ind]
Y = Y[ind]
x_trn = X[:n_trn]
y_trn = Y[:n_trn]
x_tst = X[n_trn:]
y_tst = Y[n_trn:]

res, ms = compare_k(x_trn, y_trn, x_tst, y_tst,1,201,20)