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
import datetime
import string

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
    spitData(run_type, Paths.fu_v2, needRebuild)
    # Get result for avg data
    spitData(run_type, Paths.data_avg, needRebuild)
    # Get result for linear data
    spitData(run_type, Paths.data_linear, needRebuild)
    #
    showParetoChart(results1, 1)
    showParetoChart(results2, 2)
    showParetoChart(results3, 3)
    showParetoChart(results4, 4)

    buildReportDataFrame()

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

def buildReportDataFrame():
    # Prepare data
    names = ["Naive Bayes", "Decision Tree", "K- Nearest Neighbors", "SVM", "Logistic Regression",
          "Neural Network", "LDA", "QDA", "AdaBoost"]

    runTypes = ["Normal", "Cloned", "Cluster"]

    preprocessTypes = ["Raw", "AVG", "Linear Regression"]

    showTypes = ["Confusion Matrix", "Chart"]

    alphabetList = list(string.ascii_uppercase)

    # Create exel file
    today = datetime.date.today()
    workbook = xlsxwriter.Workbook('report_' + today.strftime('%Y%m%d') + '.xlsx')

    fillDataForReportFile(workbook, 1, names, runTypes, preprocessTypes, showTypes, alphabetList, results1)
    fillDataForReportFile(workbook, 2, names, runTypes, preprocessTypes, showTypes, alphabetList, results2)
    fillDataForReportFile(workbook, 3, names, runTypes, preprocessTypes, showTypes, alphabetList, results3)
    fillDataForReportFile(workbook, 4, names, runTypes, preprocessTypes, showTypes, alphabetList, results4)

    workbook.close()

def fillDataForReportFile(workbook, i, names, runTypes, preprocessTypes, showTypes, alphabetList, results):
    # Add sheet
    worksheet = workbook.add_worksheet("Part " + str(i))

    i = i - 1
    # Build template for report
    # Create a format to use in the merged range.
    merge_format = workbook.add_format({
        'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'})

    bold_format = workbook.add_format({'bold': 1, 'border': 1})

    normal_format = workbook.add_format({
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'})

    # Set size for rows and cols
    worksheet.set_row(0, 50)
    worksheet.set_column(0, 1, 20)

    # Build template for cols
    count = 2
    countTemp = 2
    for method in names:
        # Merge 2 cells.
        worksheet.merge_range("%s1:%s1" % (alphabetList[count], alphabetList[count+1]), method, merge_format)
        for showType in showTypes:
            worksheet.set_column(count, count, 30)
            worksheet.set_column(count+1, count+1, 45)
            worksheet.write("%s2" % alphabetList[countTemp], showType, merge_format)
            countTemp += 1
        count += 2

    # Build template for rows
    count = 3
    countTemp = 3
    for runType in runTypes:
        worksheet.merge_range("A%d:A%d" % (count, count + 2), runType, merge_format)
        count += 3
        for preprocessType in preprocessTypes:
            worksheet.set_row(countTemp - 1, 250)
            worksheet.write("B%d" % countTemp, preprocessType, merge_format)
            countTemp += 1

    image_width = 800.0
    image_height = 600.0

    image_cell_width = 330.0
    image_cell_height = 247.5
    x_scale = float(image_cell_width)/float(image_width)
    y_scale = float(image_cell_height)/float(image_height)

    # Start fill results
    count = 2
    for result in results:
        for clf_descr, confusion_matrix, title in result:
            matrix = "a\tb\t<-- classified as \n"
            matrix = matrix + str(confusion_matrix[0][0]) + "\t" + str(confusion_matrix[0][1]) + "\t |    a = 0 \n"
            matrix = matrix + str(confusion_matrix[1][0]) + "\t" + str(confusion_matrix[1][1]) + "\t |    b = 1 \n"
            row, imagePath = getResultType(clf_descr, i, title)

            if count > 19:
                count = 2

            # We dont have result for SVM of cluster
            if (row == 9 or row == 10 or row == 11) and title == 'Logistic Regression':
                count += 2

            worksheet.write("%s%d" % (alphabetList[count], row), matrix, normal_format)
            worksheet.insert_image("%s%d" % (alphabetList[count + 1], row), imagePath, {'x_scale': x_scale, 'y_scale': y_scale})
            count += 2

def getResultType(clf_descr, i, title):
    if "fu_v2" in clf_descr and "normal" in clf_descr:
        return 3, getImagePath(Paths.fu_v2, "normal", i, title)
    elif "avg" in clf_descr and "normal" in clf_descr:
        return 4, getImagePath(Paths.data_avg, "normal", i, title)
    elif "linear" in clf_descr and "normal" in clf_descr:
        return 5, getImagePath(Paths.data_linear, "normal", i, title)
    elif "fu_v2" in clf_descr and "clone" in clf_descr:
        return 6, getImagePath(Paths.fu_v2, "cloned", i, title)
    elif "avg" in clf_descr and "clone" in clf_descr:
        return 7, getImagePath(Paths.data_avg, "cloned", i, title)
    elif "linear" in clf_descr and "clone" in clf_descr:
        return 8, getImagePath(Paths.data_linear, "cloned", i, title)
    elif "fu_v2" in clf_descr and "cluster" in clf_descr:
        return 9, getImagePath(Paths.fu_v2, "cluster", i, title)
    elif "avg" in clf_descr and "cluster" in clf_descr:
        return 10, getImagePath(Paths.data_avg, "cluster", i, title)
    elif "linear" in clf_descr and "cluster" in clf_descr:
        return 11, getImagePath(Paths.data_linear, "cluster", i, title)
