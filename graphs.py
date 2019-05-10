import math
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import Platt
import pandas as pd

plt.rcParams["figure.figsize"] = [10, 5]
epsilon_range = np.arange(0.002, 0.2002, 0.002)


def sigmoid(x, A, B):
    return 1 / (1 + np.exp(A * x + B))


def getSize(name):
    pcs = name.split("_")
    return int(pcs[0])


def showBestEpsilons(path, savePath, is200=True):
    """if path.startswith("files200/0.2"):
        epsilon_range = np.arange(0.002, 0.302, 0.002)
        y_telg = [0, 0.31]
    elif path.startswith("files200/0.1"):
        epsilon_range = np.arange(0.002, 0.402, 0.002)
        y_telg = [0, 0.41]
    else:
        epsilon_range = np.arange(0.002, 0.2002, 0.002)"""
    y_telg = [0, 0.21]

    if is200:
        step = 5
        size = 200
    else:
        step = 10
        size = 100

    allmeans = []
    mean_platt_losses = []
    mean_not_platt_losses = []
    name = path.split("/")[1]

    files = sorted(os.listdir(path), key=getSize)

    if len(files) != ((size - 10) / step + 1):
        size = len(files) * step

    for f in files:
        with open(path + f, "rb") as file:
            losses = pickle.load(file)
            platt_losses = pickle.load(file)

        lowest_loss = 1
        means = []
        mean_platt_losses.append(np.mean(platt_losses[1]))
        mean_not_platt_losses.append(np.mean(platt_losses[0]))

        for index, i in enumerate(epsilon_range):
            loss = np.mean(losses[index])
            means.append(loss)
            if loss < lowest_loss:
                lowest_loss = loss

        allmeans.append(means)

    """for index, mean in enumerate(allmeans):
        plt.plot(epsilon_range, mean)
        idx = mean.index(np.min(mean))
        plt.scatter(epsilon_range[idx], np.min(mean))
    plt.show()
    """

    pcs = name.split("_")
    dist = float(pcs[1])
    punktid = []

    mean_best_epsilon_losses = []

    best_epsilons = []
    platt_epsilons = []

    for index, mean in enumerate(allmeans):
        index = (index + 1) * step
        if is200:
            index += 5

        mean_best_epsilon_losses.append(np.min(mean))
        epsilon = epsilon_range[mean.index(np.min(mean))]
        best_epsilons.append(epsilon)

        platt = 1 / (index * dist + 2)
        platt_epsilons.append(platt)

        plt.scatter(index, epsilon, color="royalblue", s=15)
        plt.scatter(index, platt, color="darkorange", s=15)
        punktid.append(abs(epsilon - platt))

    title = "Keskväärtus = " + pcs[0] + " Klasside suuruste suhe = " + pcs[1]

    plt.title("Leitud parim ɛ ja Platti valitud ɛ\n" + title)
    axes = plt.gca()
    axes.set_ylim(y_telg)
    legends = ["Treenimisel leitud parim ɛ", "Platti valitud ɛ"]
    plt.legend(legends)
    plt.xticks(np.arange(10, size + step, 10))
    plt.savefig(savePath + name + ".png")
    plt.close()

    plt.plot(
        np.arange(10, size + step, step),
        punktid,
        label="abs(leitud parim ɛ - Platti valitud ɛ)",
        color="royalblue",
    )
    plt.title("Leitud ɛ ja Platti ɛ suhe\n" + title)
    axes = plt.gca()
    axes.set_ylim([0, 0.3])
    plt.legend()
    plt.xticks(np.arange(10, size + step, 10))
    plt.savefig(savePath + name + "_distance.png")
    plt.close()

    plt.plot(
        np.arange(10, size + step, step),
        mean_platt_losses,
        label="Keskmine kadu Platti ɛ",
        color="darkorange",
    )
    plt.plot(
        np.arange(10, size + step, step),
        mean_best_epsilon_losses,
        label="Keskmine kadu leitud parima ɛ",
        color="royalblue",
    )
    plt.plot(
        np.arange(10, size + step, step),
        mean_not_platt_losses,
        label="Keskmine kadu leitud ilma Platti skaleerimiseta",
        color="black",
    )
    plt.title("Parima ɛ ja Platti ɛ logistilised kaod\n" + title)
    axes = plt.gca()
    # axes.set_ylim([0.3, 0.8])
    plt.legend()
    plt.xticks(np.arange(10, size + step, 10))
    plt.savefig(savePath + name + "_platt.png")
    plt.close()

    return best_epsilons, platt_epsilons, mean_best_epsilon_losses, mean_platt_losses


def showAllGraphs(path):
    files = sorted(os.listdir(path), key=getSize)

    thetas_file = files[0]

    with open(path + thetas_file, "rb") as file:
        pickle.load(file)
        pickle.load(file)
        thetas = pickle.load(file)
    
    mu = float(thetas_file.split("_")[1])

    withoutPlatt = []
    withPlatt = []

    points = np.arange(-3, 3, 0.01)

    for index, row in thetas.iterrows():
        withoutPlatt.append(Platt.sigmoid(points, row[0], row[1]))
        withPlatt.append(Platt.sigmoid(points, row[2], row[3]))

    withoutPlatt = pd.DataFrame(withoutPlatt)
    withPlatt = pd.DataFrame(withPlatt)

    meanPointsWithout = []
    meanPointsWith = []
    for index, point in enumerate(points):
        meanPointsWithout.append(np.mean(withoutPlatt[index]))
        meanPointsWith.append(np.mean(withPlatt[index]))

    plt.plot(points, meanPointsWithout, label="Ilma märgendite silumiseta")
    plt.plot(points, meanPointsWith, label="Platti valemiga märgendeid siludes")
    plt.plot(points, Platt.sigmoid(points, -(mu*2), 0), label="Bayesi-optimaalne", color="black")

    plt.legend()
    plt.savefig(str(mu)+"_plattvs.png")
    plt.show()


def saveAllGraphs(path):
    best_epsilons = {"1.0": [], "1.5": [], "2.0": []}
    platt_epsilons = {"1.0": [], "1.5": [], "2.0": []}
    best_losses = {"1.0": [], "1.5": [], "2.0": []}
    platt_losses = {"1.0": [], "1.5": [], "2.0": []}

    if not path.endswith("/"):
        path = path + "/"
    if path.startswith("files200"):
        savePath = "pictures200/"
        is200 = True
    else:
        savePath = "pictures/"
        is200 = False
    for dir in os.listdir(path):
        if dir.startswith("0.1") or dir.startswith("0.2"):
            continue
        newpath = path + dir + "/"
        print(newpath)
        dir = dir[-3:]

        best, platt, best_loss, platt_loss = showBestEpsilons(newpath, savePath, is200)
        best_epsilons[dir].append(best)
        platt_epsilons[dir] = platt
        best_losses[dir].append(best_loss)
        platt_losses[dir].append(platt_loss)

    for key, value in best_epsilons.items():
        if key == "1.0":
            saveErrorRateGraph(key, value)
        saveTotalEpsilonGraph(key, value, platt_epsilons[key])

    for key, value in best_losses.items():
        saveTotalLossDiffGraph(key, value, platt_losses[key])



def errorRate(x_train, y_train, vs):
    error_rate = 0
    for index, x in enumerate(x_train):
        if x < vs and y_train[index] == 1:
            error_rate += 1
        if x >= vs and y_train[index] == 0:
            error_rate += 1

    return error_rate / len(y_train)


def getErrorRate(mu, size):
    allrates = []
    for i in range(10000):
        pos = np.random.normal(mu, 1, size)
        neg = np.random.normal(-mu, 1, size)

        x_train = np.append(pos, neg)
        y_train = [1] * size + [0] * size

        error_rate = errorRate(x_train, y_train, 0)
        allrates.append(error_rate)
    return np.mean(allrates)

def getErrorRatesForMus(mus, size):
    means = []
    for mu in mus:
        rate = getErrorRate(mu,size)
        means.append(rate)
    return means


def saveErrorRateGraph(key, value):
    colors = [
        "greenyellow",
        "lightgreen",
        "limegreen",
        "turquoise",
        "cadetblue",
        "lightskyblue",
        "steelblue",
        "navy",
        "darkblue",
        "midnightblue"
    ]

    mus = [0.5, 1.0, 1.5, 2.0]

    sizeIndex = np.arange(0,40,4)

    val = pd.DataFrame(value)

    sizes = []
    allrates = []
    alleps = []

    for index, sizeidx in enumerate(sizeIndex):
        size = (sizeidx + 1) *5 + 5
        sizes += [size]*len(mus)

        rates = getErrorRatesForMus(mus,size)
        allrates += rates

        eps = val.iloc[:][sizeidx]
        alleps += eps.tolist()

        print(rates,eps)
        plt.plot(rates,eps,color=colors[index], label="Suurus "+str(size))

    plt.xlabel("veamäär")
    plt.ylabel("silumismäär")
    plt.legend()
    #plt.show()
    
    x_train = np.array([allrates,sizes]).T
    y_train = np.array(alleps).T
    linreg = LinearRegression().fit(x_train, y_train)
    
    #testSizes = np.delete(np.arange(0,40,1),sizeIndex)
    testSizes = [0,1,2]
    x_test_size = []
    x_test_rates = []

    y_test = np.array([])

    for index,size in enumerate(testSizes):
        size = (sizeidx + 1) *5 + 5
        x_test_size += [size]*len(mus)

        rates = getErrorRatesForMus(mus,size)
        x_test_rates += rates

        eps = val.iloc[:][sizeidx]
        print(eps)
        y_test = np.vstack((y_test,eps))
    
    x_test = np.array([x_test_rates, x_test_size]).T
    #y_test = np.asarray(y_test,dtype=np.float64)
    print(y_test)

    preds = linreg.predict(x_test)
    print(preds[:10], y_test[:10])
    #print(Platt.log_loss(y_test,preds))

def saveTotalEpsilonGraph(key, value, platt_epsilon):
    colors = [
        "limegreen",
        "turquoise",
        "cadetblue",
        "lightskyblue",
        "steelblue",
        "navy",
    ]
    mus = [0.5, 1.0, 1.5, 2.0]

    for index, val in enumerate(value):
        plt.plot(
            np.arange(10, 205, 5),
            val,
            color=colors[index]
            ,label="Veamäär = " + str(mus[index])
        )
    plt.plot(
        np.arange(10, 205, 5), platt_epsilon, color="black", label="Platti silumismäär"
    )

    axes = plt.gca()
    axes.set_ylim([-0.001, 0.175])
    plt.xlabel("Positiivse klassi suurus")
    plt.ylabel("Silumismäär")
    plt.xticks(np.arange(10, 205, 10))
    plt.legend(loc=1)

    plt.savefig(key + "_epsilons.png")
    plt.close()

saveAllGraphs("files200/")

def saveTotalLossDiffGraph(key, value, platt_losses):
    colors = [
        "limegreen",
        "turquoise",
        "cadetblue",
        "lightskyblue",
        "steelblue",
        "navy",
    ]
    mus = [0.5, 1.0, 1.5, 2.0]

    for index, val in enumerate(value):
        plt.plot(
            np.arange(10, 205, 5),
            [platt_losses[index][i] - j for i, j in enumerate(val)],
            color=colors[index],
            label="μ = " + str(mus[index]),
        )
    axes = plt.gca()
    axes.set_ylim([-0.001, 0.09])
    plt.xticks(np.arange(10, 205, 10))
    plt.legend(loc=1)
    plt.savefig(key + "_vs.png")
    plt.close()


if __name__ == "__main__":
    args = sys.argv

    if args[1] == "showGraph":
        showAllGraphs("files/0.5_1.0/")
    else:
        assert len(args) == 2
        saveAllGraphs(args[1])
