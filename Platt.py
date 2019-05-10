import os
import pickle
import sys
from math import log

import numpy as np
import pandas as pd
from scipy.optimize import fmin_bfgs
from sklearn.metrics import log_loss
from sklearn.utils import column_or_1d

epsilon_range = np.arange(0.002, 0.2002, 0.002)


def logReg(df, y, isPlatt, sample_weight=None, epsilon=None):
    df = column_or_1d(df)
    y = column_or_1d(y)

    F = df  # F follows Platt's notations
    tiny = np.finfo(np.float).tiny  # to avoid division by 0 warning

    # Bayesian priors (see Platt end of section 2.2)
    prior0 = float(np.sum(y <= 0))
    prior1 = y.shape[0] - prior0

    if isPlatt:
        T = np.zeros(y.shape)
        if epsilon is None:
            T[y > 0] = (prior1 + 1.0) / (prior1 + 2.0)
            T[y <= 0] = 1.0 / (prior0 + 2.0)
        else:
            T[y > 0] = 1 - epsilon
            T[y <= 0] = epsilon
    else:
        T = y

    T1 = 1.0 - T

    def objective(AB):
        # From Platt (beginning of Section 2.2)
        E = np.exp(AB[0] * F + AB[1])
        P = 1.0 / (1.0 + E)
        l = -(T * np.log(P + tiny) + T1 * np.log(1.0 - P + tiny))
        if sample_weight is not None:
            return (sample_weight * l).sum()
        else:
            return l.sum()

    def grad(AB):
        # gradient of the objective function
        E = np.exp(AB[0] * F + AB[1])
        P = 1.0 / (1.0 + E)
        TEP_minus_T1P = P * (T * E - T1)
        if sample_weight is not None:
            TEP_minus_T1P *= sample_weight
        dA = np.dot(TEP_minus_T1P, F)
        dB = np.sum(TEP_minus_T1P)
        return np.array([dA, dB])

    AB0 = np.array([0.0, log((prior0 + 1.0) / (prior1 + 1.0))])
    AB_ = fmin_bfgs(objective, AB0, fprime=grad, disp=False)
    return AB_


def sigmoid(x, A, B):
    return 1 / (1 + np.exp(A * x + B))


def logloss(xTest, yTest, theta):
    pred = sigmoid(xTest, theta[0], theta[1])
    return log_loss(yTest, pred)


def lossesForOneIteration(size, mu, dist):
    positives = np.random.normal(mu, 1, size)
    negatives = np.random.normal(-mu, 1, int(size * dist))
    points = np.append(negatives, positives)
    labels = [0] * len(negatives) + [1] * len(positives)

    thetaWithoutPlatt = logReg(points, labels, isPlatt=False)
    thetaWithPlatt = logReg(points, labels, isPlatt=True)

    lossWithoutPlatt = logloss(xTest, yTest, thetaWithoutPlatt)
    lossWithPlatt = logloss(xTest, yTest, thetaWithPlatt)

    losses = []
    for i in epsilon_range:
        theta = logReg(points, labels, isPlatt=True, epsilon=i)
        log_loss = logloss(xTest, yTest, theta)
        losses.append(log_loss)

    return (
        np.concatenate([thetaWithoutPlatt, thetaWithPlatt]),
        [lossWithoutPlatt, lossWithPlatt],
        losses,
    )


def findAllLossesForEpsilons(iters, size, mu, dist):
    thetas = []
    platt_losses = []
    losses = []

    for i in range(iters):
        if i % 300 == 0:
            print(i)

        theta, platt_loss, loss = lossesForOneIteration(size, mu, dist)
        losses.append(loss)
        thetas.append(theta)
        platt_losses.append(platt_loss)

    thetas = pd.DataFrame(thetas)
    platt_losses = pd.DataFrame(platt_losses)
    losses = pd.DataFrame(losses)

    folder = "files/" + str(mu) + "_" + str(dist) + "/"
    ext = ".txt"
    if not os.path.isdir(folder):
        os.mkdir(folder)

    with open(
        folder + str(size) + "_" + str(mu) + "_" + str(dist) + ext, "wb+"
    ) as file:
        pickle.dump(losses, file)
        pickle.dump(platt_losses, file)
        pickle.dump(thetas, file)

    return losses


def createTest(mu, testSize):
    PosTest = np.random.normal(mu, 1, testSize)
    NegTest = np.random.normal(-mu, 1, testSize)
    xTest = np.append(PosTest, NegTest)
    yTest = [1.0] * testSize + [0.0] * testSize
    return xTest, yTest


if __name__ == "__main__":
    args = sys.argv
    size_range = np.arange(10, 205, 5)

    if args[1] == "all":
        iters = 1000
        size = size_range
        mu = [0.1, 0.2, 0.5, 0.5, 0.5, 1.0, 1.5, 2.0]
        dist = [1.0, 1.0, 1.0, 1.5, 2.0, 1.0, 1.0, 1.0]

        for index in range(len(mu)):
            iter_mu = mu[index]
            iter_dist = dist[index]
            path = "files/" + str(iter_mu) + "_" + str(iter_dist) + "/"

            xTest, yTest = createTest(iter_mu, 10000)

            for i in size:
                findAllLossesForEpsilons(iters, i, iter_mu, iter_dist)

    else:
        assert len(args) == 5
        iters = int(args[1])

        if args[2] == "all":
            size = size_range
        else:
            size = [int(args[2])]

        mu = float(args[3])
        dist = float(args[4])

        path = "files/" + str(mu) + "_" + str(dist) + "/"

        xTest, yTest = createTest(mu, 10000)

        for i in size:
            findAllLossesForEpsilons(iters, i, mu, dist)
