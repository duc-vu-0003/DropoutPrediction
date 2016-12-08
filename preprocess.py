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
from matplotlib.pyplot import cm
import scipy.spatial.distance as dist
from operator import itemgetter
import xlsxwriter

results1 = []
results2 = []
results3 = []
results4 = []

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
def run(run_type, needRebuild = False):
    # Init result array
    del results1[:]
    del results2[:]
    del results3[:]
    del results4[:]

    # Get result for raw data
    # spitData(run_type, Paths.fu_v2, needRebuild)
    # Get result for avg data
    # spitData(run_type, Paths.data_avg, needRebuild)
    # Get result for linear data
    spitData(run_type, Paths.data_linear, needRebuild)

    showParetoChart(results1, 1)
    showParetoChart(results2, 2)
    showParetoChart(results3, 3)
    showParetoChart(results4, 4)

    buildReportDataFrame(results1, 1)
    buildReportDataFrame(results2, 2)
    buildReportDataFrame(results3, 3)
    buildReportDataFrame(results4, 4)

def showParetoChart(results, i):
    plt.title("Result for part " + str(i))
    colormap = cm.gist_ncar
    # plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, 50)])
    Xs = []
    Ys = []

    labels = []
    dotItems = []

    figure = plt.gcf() # get current figure
    figure.set_size_inches(6, 8)
    for result in results:
        for clf_descr, confusion_matrix, title in result:
            tn, fp, fn, tp = confusion_matrix.ravel()
            drop_acc = 0
            nondrop_acc = 0
            if tp > 0:
                drop_acc = float(tp) / float((tp + fn))

            if tn > 0:
                nondrop_acc = float(tn) / float(tn + fp)

            labels.append(clf_descr)
            Xs.append(nondrop_acc * 100)
            Ys.append(drop_acc * 100)

    # Find lowest values for drop_acc and highest for nondrop_acc
    p_front = pareto_frontier(Ys, Xs, maxX = True, maxY = True)
    # Plot a scatter graph of all results
    colors = cm.rainbow(np.linspace(0, 1, len(Ys)))
    for j in range(0, len(Xs)):
        labelItem = str(j) + " | " + labels[j]
        dotItems.append(plt.scatter(Ys[j], Xs[j], label=labelItem, color=colors[j]))

        for z in range(0, len(p_front[0])):
            if p_front[0][z] == Ys[j] and p_front[1][z] == Xs[j]:
                plt.annotate(j, (Ys[j], Xs[j]))

    # Then plot the Pareto frontier on top
    plt.plot(p_front[0], p_front[1], '--', label='Pareto Frontier', alpha=0.5)

    plt.xlabel('Drop Predict Accuracy (%)')
    plt.ylabel('Non-drop Predict Accuracy (%)')
    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, fontsize='small')
    savePath = Paths.report_path + "/" + "result" + str(i) + ".png"
    saveDir = os.path.dirname(savePath)
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(savePath, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=600) #bbox_extra_artists=(lgd,),
    plt.close()

def pareto_frontier(Xs, Ys, maxX = True, maxY = True):
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    p_front = [myList[0]]
    for pair in myList[1:]:
        if maxY:
            if pair[0] == p_front[-1][0] and pair[1] >= p_front[-1][1]:
                p_front.pop(-1)
                p_front.append(pair)
            elif pair[1] >= p_front[-1][1]:
                p_front.append(pair)
        else:
            if pair[0] == p_front[-1][0] and pair[1] <= p_front[-1][1]:
                p_front.pop(-1)
                p_front.append(pair)
            elif pair[1] <= p_front[-1][1]:
                p_front.append(pair)

    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY

# We will spit data to 4 folds, each fold will have same numbers of data, same dropout numbers.
# One for test
# One for normal training
# One for clone
# One for cluster
# We will run 4 times and change data to test each time we run.
# Type 0 = normal
# Type 1 = cloned
# Type 2 = cluster
# Type 3 = Run All
def spitData(run_type, dataPath, needRebuild):
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0) #random_state=None will return diffirent fold everytime
    dataFUV2 = []
    if os.path.exists(dataPath):
        dataFUV2 = pd.read_csv(dataPath, sep=",", encoding='utf-8', low_memory=False)

    if dataPath == Paths.fu_v2:
        dataFUV2 = dataFUV2.fillna(method='ffill')

    print("--------------------------------------------")
    print("--------------------------------------------")
    print("Processing: " + dataPath)
    print("--------------------------------------------")
    print("--------------------------------------------")

    X = dataFUV2.ix[:,1:].values
    y = dataFUV2.ix[:,0]

    # # normalize the data attributes
    # normalized_X = preprocessing.normalize(X)
    # # standardize the data attributes
    # standardized_X = preprocessing.scale(X)

    count = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if run_type == 0:
            appendData(count, normalData(X_train, X_test, y_train, y_test, dataFUV2.ix[:,1:].columns.values.tolist(), dataPath, "normal", count, needRebuild))
        elif run_type == 1:
            print("--------------------------------------------")
            print("Run with cloned data:")
            appendData(count, clonedData(X_train, X_test, y_train, y_test, dataFUV2.ix[:,1:].columns.values.tolist(), dataPath, "cloned", count, needRebuild))
            print("--------------------------------------------")
        elif run_type == 2:
            print("--------------------------------------------")
            print("Run with cluster data:")
            appendData(count, clusterData(X_train, X_test, y_train, y_test, dataFUV2.ix[:,1:].columns.values.tolist(), dataPath, "cluster", count, needRebuild))
            print("--------------------------------------------")
        elif run_type == 3:
            appendData(count, normalData(X_train, X_test, y_train, y_test, dataFUV2.ix[:,1:].columns.values.tolist(), dataPath, "normal", count, needRebuild))
            print("--------------------------------------------")
            print("Run with cloned data:")
            appendData(count, clonedData(X_train, X_test, y_train, y_test, dataFUV2.ix[:,1:].columns.values.tolist(), dataPath, "cloned", count, needRebuild))
            print("--------------------------------------------")
            print("Run with cluster data:")
            appendData(count, clusterData(X_train, X_test, y_train, y_test, dataFUV2.ix[:,1:].columns.values.tolist(), dataPath, "cluster", count, needRebuild))
            print("--------------------------------------------")

        count += 1

def appendData(dataPart, results):
    if dataPart == 0:
        results1.append(results)
    elif dataPart == 1:
        results2.append(results)
    elif dataPart == 2:
        results3.append(results)
    elif dataPart == 3:
        results4.append(results)

def itemfreq(a):
    items, inv = np.unique(a, return_inverse=True)
    freq = np.bincount(inv)
    print(freq)

# Run with raw data
def normalData(X_train, X_test, y_train, y_test, target_names, dataPath, runType, dataPart, needRebuild):
    print("--------------------------------------------")
    print("Run with normal data:")
    return classifiers(X_train, X_test, y_train, y_test, target_names, dataPath, runType, dataPart, needRebuild)
    print("--------------------------------------------")

# With this technique, first we spit data to 2 parts
# One for dropout
# One for non dropout (This part use to cluster to n small parts)
def clusterData(X_train, X_test, y_train, y_test, target_names, dataPath, runType, dataPart, needRebuild):
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
    return classifiersMultiLabel(X_train, X_test, y_train, y_test, target_names, dataPath, runType, dataPart, needRebuild)

# This is one way to overcome unbalanced data problem
# We will cloned data of dropout student as soon as drop = non-drop
def clonedData(X_train, X_test, y_train, y_test, target_names, dataPath, runType, dataPart, needRebuild):
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

    return classifiers(X_train, X_test, y_train, y_test, target_names, dataPath, runType, dataPart, needRebuild)

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

def buildReportDataFrame(results, i):
    # Create report columns
    columns = ['Name','Confusion Matrix']
    dfReport = pd.DataFrame(columns=columns)
    for result in results:
        for clf_descr, confusion_matrix, title in result:
            matrix = str(confusion_matrix[0][0]) + "\t" + str(confusion_matrix[0][1]) + "\n" + str(confusion_matrix[1][0]) + "\t" + str(confusion_matrix[1][1])
            print("clf_descr " + " " + matrix)
            dfReport = dfReport.append({'Name': clf_descr, 'Confusion Matrix': matrix }, ignore_index=True)

    dfReport.to_excel('report.xlsx', sheet_name='sheet' + str(i), index=False)
