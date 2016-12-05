import random
import os
import pickle
import pandas as pd
import numpy as np
import logging
from utils import Paths
import time
import collections
import encodings
import re
from os import path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer
from sklearn.model_selection import StratifiedKFold
from classification import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
from operator import itemgetter

# This method use to fill missing values
# We will overcome missing values with 2 techniques
# Use avg values by Imputer of sklearn.preprocessing
# Predict with LinearRegression
def predictScore():
    dataFUV2 = []
    if os.path.exists(Paths.fu_v2):
        dataFUV2 = pd.read_csv(Paths.fu_v2, sep=",", encoding='utf-8', low_memory=False)

    # Get all the columns from the dataframe.
    columns = dataFUV2.columns.tolist()
    # Filter the columns to remove ones we don't want.
    columns = [c for c in columns if c not in ["EntryGrade (group)"]]
    # Store the variable we'll be predicting on.
    target = "EntryGrade (group)"

    colsToDrop = ['EntryGrade (group)']
    dataFUV2NotGrade = dataFUV2.drop(colsToDrop, axis=1)
    dataGrade = pd.DataFrame(dataFUV2, columns=colsToDrop)

    # Fill missing data
    fill_NaN = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputed_DF = pd.DataFrame(fill_NaN.fit_transform(dataFUV2NotGrade))
    imputed_DF.columns = dataFUV2NotGrade.columns
    imputed_DF.index = dataFUV2NotGrade.index

    imputed_DF['EntryGrade (group)'] = dataGrade

    # Get Training set (include all students has input score)
    # Get Test set (include all students missing score)
    dataFUV2Train = imputed_DF[imputed_DF["EntryGrade (group)"] > 0]
    dataFUV2Test = imputed_DF[(imputed_DF["EntryGrade (group)"] > 0) == False]

    print("dataFUV2Train.shape: " + str(dataFUV2Train.shape))
    print("dataFUV2Test.shape: " + str(dataFUV2Test.shape))

    # Initialize the model class.
    model = LinearRegression()
    # Fit the model to the training data.
    model.fit(dataFUV2Train[columns], dataFUV2Train[target])
    # Generate our predictions for the test set.
    predictions = model.predict(dataFUV2Test[columns])
    dataFUV2Test[target] = predictions

    # Join training set and test set
    finalDataFrame = pd.concat([dataFUV2Train, dataFUV2Test])
    # Write result to file
    finalDataFrame.to_csv(Paths.data_linear, index=False)

    # Fill missing values with avg values
    dataAVGFU = pd.DataFrame(fill_NaN.fit_transform(dataFUV2))
    dataAVGFU.columns = dataFUV2.columns
    dataAVGFU.index = dataFUV2.index
    # Write result to file
    dataAVGFU.to_csv(Paths.data_avg, index=False)

# Run with 3 data sets
# 1 - Raw data
# 2 - Handler missing data with AVG values
# 3 - Handler missing data by linear regresstion
def run(run_type):
    results1 = []
    results2 = []
    results3 = []
    results4 = []

    results1Raw, results2Raw, results3Raw, results4Raw = spitData(run_type, Paths.fu_v2)
    # results1Avg, results2Avg, results3Avg, results4Avg = spitData(run_type, Paths.data_avg)
    # results1Linear, results2Linear, results3Linear, results4Linear = spitData(run_type, Paths.data_linear)

    # results1.append(results1Raw)
    # results1.append(results1Avg)
    # results1.append(results1Linear)

    # results2.append(results2Raw)
    # results2.append(results2Avg)
    # results2.append(results2Linear)

    # results3.append(results3Raw)
    # results3.append(results3Avg)
    # results3.append(results3Linear)

    # results4.append(results4Raw)
    # results4.append(results4Avg)
    # results4.append(results4Linear)

    # plt.title("Result")
    # for clf_descr, confusion_matrix in results1[0]:
    #     # plot the data observations
    #     plt.plot(confusion_matrix[0][0],confusion_matrix[1][1],'o')
    #     # # plot the centroids
    #     # lines = plt.plot(centroids[i,0],centroids[i,1],'kx')
    #     # # make the centroid x's bigger
    #     # plt.setp(lines,ms=15.0)
    #     # plt.setp(lines,mew=2.0)
    # plt.show()


# We will spit data to 4 folds, each fold will have same numbers of data, same dropout numbers.
# One for test
# One for normal training
# One for clone
# One for cluster
# We will run 4 times and change data to test each time we run.
# Type 0 = normal
# Type 1 = cloned
# Type 2 = cluster
def spitData(run_type, dataPath):
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0) #random_state=None will return diffirent fold everytime
    dataFUV2 = []
    if os.path.exists(dataPath):
        dataFUV2 = pd.read_csv(dataPath, sep=",", encoding='utf-8', low_memory=False)

    if dataPath == Paths.fu_v2:
        dataFUV2 = dataFUV2.fillna(method='ffill')

    X = dataFUV2.ix[:,1:].values
    y = dataFUV2.ix[:,0]

    # # normalize the data attributes
    # normalized_X = preprocessing.normalize(X)
    # # standardize the data attributes
    # standardized_X = preprocessing.scale(X)

    results1 = []
    results2 = []
    results3 = []
    results4 = []

    count = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        results = []
        if run_type == 0:
            results = normalData(X_train, X_test, y_train, y_test, dataFUV2.ix[:,1:].columns.values.tolist(), dataPath, "normal", count)
            if count == 0:
                results1.append(results)
            elif count == 1:
                results2.append(results)
            elif count == 2:
                results3.append(results)
            else:
                results4.append(results)
        elif run_type == 1:
            print("--------------------------------------------")
            print("Run with cloned data:")
            results = clonedData(X_train, X_test, y_train, y_test, dataFUV2.ix[:,1:].columns.values.tolist(), dataPath, "cloned", count)
            if count == 0:
                results1.append(results)
            elif count == 1:
                results2.append(results)
            elif count == 2:
                results3.append(results)
            else:
                results4.append(results)
            print("--------------------------------------------")
        elif run_type == 2:
            print("--------------------------------------------")
            print("Run with cluster data:")
            results = clusterData(X_train, X_test, y_train, y_test, dataFUV2.ix[:,1:].columns.values.tolist(), dataPath, "cluster", count)
            if count == 0:
                results1.append(results)
            elif count == 1:
                results2.append(results)
            elif count == 2:
                results3.append(results)
            else:
                results4.append(results)
            print("--------------------------------------------")
        else:
            results = normalData(X_train, X_test, y_train, y_test, dataFUV2.ix[:,1:].columns.values.tolist(), dataPath, "normal", count)
            if count == 0:
                results1.append(results)
            elif count == 1:
                results2.append(results)
            elif count == 2:
                results3.append(results)
            else:
                results4.append(results)
            print("--------------------------------------------")
            print("Run with cloned data:")
            results = clonedData(X_train, X_test, y_train, y_test, dataFUV2.ix[:,1:].columns.values.tolist(), dataPath, "cloned", count)
            if count == 0:
                results1.append(results)
            elif count == 1:
                results2.append(results)
            elif count == 2:
                results3.append(results)
            else:
                results4.append(results)
            print("--------------------------------------------")
            print("--------------------------------------------")
            print("Run with cluster data:")
            results = clusterData(X_train, X_test, y_train, y_test, dataFUV2.ix[:,1:].columns.values.tolist(), dataPath, "cluster", count)
            if count == 0:
                results1.append(results)
            elif count == 1:
                results2.append(results)
            elif count == 2:
                results3.append(results)
            else:
                results4.append(results)
            print("--------------------------------------------")

        count += 1
    return results1, results2, results3, results4

def itemfreq(a):
    items, inv = np.unique(a, return_inverse=True)
    freq = np.bincount(inv)
    print(freq)

# Run with raw data
def normalData(X_train, X_test, y_train, y_test, target_names, dataPath, runType, dataPart):
    print("--------------------------------------------")
    print("Run with normal data:")
    return classifiers(X_train, X_test, y_train, y_test, target_names, dataPath, runType, dataPart)
    print("--------------------------------------------")

# With this technique, first we spit data to 2 parts
# One for dropout
# One for non dropout (This part use to cluster to n small parts)
def clusterData(X_train, X_test, y_train, y_test, target_names, dataPath, runType, dataPart):
    dropOutPos, nondropOutPos = getDropOutPosition(y_train)
    dropOutCount = len(dropOutPos)
    nonDropOutCount = len(y_train) - dropOutCount
    estimate_cluster = nonDropOutCount / dropOutCount
    if dropOutCount*estimate_cluster < nonDropOutCount:
        estimate_cluster+=1
    print("Estimate cluster number: " + str(estimate_cluster))

    # We will get non-drop list for cluster
    X_train_non_drop = np.zeros(shape=(nonDropOutCount,18))
    count = 0
    for i in nondropOutPos:
        X_train_non_drop[count] = X_train[i]
        count+=1

    # Get dropout list from traning set
    X_train_drop = np.zeros(shape=(dropOutCount,18))
    y_train_drop = np.zeros(shape=(dropOutCount,))
    count = 0
    for i in dropOutPos:
        X_train_drop[count] = X_train[i]
        y_train_drop[count] = y_train.iloc[i]
        count+=1

    y_train_drop = refineDropLabel(y_train_drop, False)

    # After get non-drop list -> run cluter to spit non-drop part to (nonDropOutCount / dropOutCount) parts
    # Because result of kmean not equal -> calculator distance of each points to center points
    # -> assign it to nearest point as soon as we have (nonDropOutCount / dropOutCount) parts equal
    print("Start cluster training set...")
    sameSizeLabels = cluster(X_train_non_drop, estimate_cluster, True, False)

    # Join cluster result to training set
    # We will join non-drop part with dropout part
    # We also join same size labels with dropout labels
    X_train = np.concatenate((X_train_non_drop, X_train_drop), axis=0)
    y_train = np.concatenate((sameSizeLabels, y_train_drop), axis=0)
    print("Finish cluster training set...")

    print("Start cluster test set...")
    return classifiersMultiLabel(X_train, X_test, y_train, y_test, target_names, dataPath, runType, dataPart)

# This is one way to overcome unbalanced data problem
# We will cloned data of dropout student as soon as drop = non-drop
def clonedData(X_train, X_test, y_train, y_test, target_names, dataPath, runType, dataPart):
    # np.concatenate((a, b), axis=0)
    # Get dropout position
    dropOutPos, nondropOutPos = getDropOutPosition(y_train)
    dropOutCount = len(dropOutPos)
    nonDropOutCount = len(y_train) - dropOutCount
    needCloneCount = nonDropOutCount - dropOutCount
    print("Before cloned data:")
    print("dropOutCount: " + str(dropOutCount))
    print("nonDropOutCount: " + str(nonDropOutCount))
    # From position -> we get list of X and y of dropout students to clone
    X_train_dropout = np.zeros(shape=(needCloneCount,18))
    y_train_dropout = np.zeros(shape=(needCloneCount,))
    count = 0
    for i in y_train_dropout:
        temp = getRealPositon(dropOutCount, count)
        realPosition = dropOutPos[temp]
        # print("temp: " + str(temp))
        # print("Real Position: " + str(realPosition))
        X_train_dropout[count] = X_train[realPosition]
        y_train_dropout[count] = y_train.iloc[realPosition]
        count+=1

    # After get this data -> clone data of dropout student as soon as drop = non-drop
    # Append cloned data to training set
    X_train = np.concatenate((X_train, X_train_dropout), axis=0)
    y_train = np.concatenate((y_train, y_train_dropout), axis=0)

    print("After cloned data:")
    dropOutPos, nondropOutPos = getDropOutPosition(y_train)
    dropOutCount = len(dropOutPos)
    nonDropOutCount = len(y_train) - dropOutCount
    print("dropOutCount: " + str(dropOutCount))
    print("nonDropOutCount: " + str(nonDropOutCount))

    return classifiers(X_train, X_test, y_train, y_test, target_names, dataPath, runType, dataPart)

def getDropOutPosition(y):
    dropOutPos = []
    nondropOutPos = []
    count = 0
    for i in y:
        if i == 1:
            dropOutPos.append(count)
        else:
            nondropOutPos.append(count)
        count+=1

    return dropOutPos, nondropOutPos

def getRealPositon(dropCount, position):
    if position < dropCount:
        return position
    elif position % dropCount > 1:
        return position % dropCount
    else:
        radio = position / dropCount
        return position - dropCount*radio

""" in: point-to-cluster-centre distances D, Npt x C
            e.g. from scipy.spatial.distance.cdist
        out: xtoc, X -> C, equal-size clusters
        method: sort all D, greedy"""
def samesizecluster(D):
    Npt, C = D.shape
    clustersize = (Npt + C - 1) // C
    xcd = list( np.ndenumerate(D) )  # ((0,0), d00), ((0,1), d01) ...
    xcd.sort( key=itemgetter(1) )
    xtoc = np.ones( Npt, int ) * -1
    nincluster = np.zeros( C, int )
    nall = 0
    for (x,c), d in xcd:
        if xtoc[x] < 0  and  nincluster[c] < clustersize:
            xtoc[x] = c + 1
            nincluster[c] += 1
            nall += 1
            if nall >= Npt:  break
    return xtoc

def cluster(X, k, needSameSize, needShowGraph):
    k_means = KMeans(init='k-means++', n_clusters=k, n_init=10)
    k_means.fit(X)
    labels = k_means.labels_
    centroids = k_means.cluster_centers_
    sameSizeLabels = []

    if needSameSize:
        print("Processing same size from cluster results...")
        # Process same size from cluster results
        distances = dist.cdist(X, centroids, "euclidean")
        sameSizeLabels = samesizecluster(distances)
        print "Samesizecluster sizes:", np.bincount(sameSizeLabels)
    else:
        sameSizeLabels = labels

    if needShowGraph:
        # Show on graphs
        for i in range(estimate_cluster):
            ds = X_train_non_drop[np.where(labels==i)]
            # plot the data observations
            plt.plot(ds[:,0],ds[:,1],'o')
            # plot the centroids
            lines = plt.plot(centroids[i,0],centroids[i,1],'kx')
            # make the centroid x's bigger
            plt.setp(lines,ms=15.0)
            plt.setp(lines,mew=2.0)
        plt.show()

    return sameSizeLabels

def writeDataFrameToText(data):
    for index, row in data.iterrows():
        if i > len(p):
           break
        else:
           f = open(str(i)+'.txt', 'w')
           f.write(row[0])
           f.close()
           i+=1
