from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import sys
import os
from time import time
from sklearn.utils.extmath import density
from sklearn import metrics
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from utils import Paths
import itertools
from sklearn.externals import joblib

# We run classify ...
def classifiers(X_train, X_test, y_train, y_test, target_names, dataPath, runType, dataPart, needRebuild):
    names = ["Naive Bayes", "Decision Tree", "K- Nearest Neighbors", "SVM", "Logistic Regression",
          "Neural Network", "LDA", "QDA", "AdaBoost"]

    classifiers = [
            GaussianNB(),
            DecisionTreeClassifier(max_depth=10),
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            LogisticRegression(),
            MLPClassifier(alpha=1),
            LinearDiscriminantAnalysis(),
            QuadraticDiscriminantAnalysis(),
            AdaBoostClassifier()]

    results = []

    # Write result
    reportPath = getReportPath(dataPath, runType, dataPart)
    dir = os.path.dirname(reportPath)
    if not os.path.exists(dir):
        os.makedirs(dir)
    # iterate over classifiers
    f = open(reportPath, 'w')
    for name, clf in zip(names, classifiers):
        f.write(name + ": \n")
        f.write("##############################################\n")
        results.append(benchmarkModel(clf, name, X_train, X_test, y_train, y_test, target_names, False, f, runType, dataPath, dataPart, needRebuild))
    f.close()

    classes = [0, 1]
    for clf_descr, confusion_matrix, title in results:
        fig = plt.figure(1)
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)
        thresh = confusion_matrix.max() / 2.
        plt.title(dataPath + "/" + runType + ": " + title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, confusion_matrix[i, j],
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        savePath = getImagePath(dataPath, runType, dataPart, title)
        saveDir = os.path.dirname(savePath)
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        plt.savefig(savePath)
        plt.close(fig)

    return results

def classifiersMultiLabel(X_train, X_test, y_train, y_test, target_names, dataPath, runType, dataPart, needRebuild):
    names = ["Naive Bayes", "Decision Tree", "K- Nearest Neighbors", #"SVM",
          "Logistic Regression", "Neural Network", "LDA", "QDA", "AdaBoost"]

    classifiers = [
            OneVsRestClassifier(GaussianNB()),
            OneVsRestClassifier(DecisionTreeClassifier(max_depth=10)),
            OneVsRestClassifier(KNeighborsClassifier(3)),
            # OneVsRestClassifier(SVC(kernel="linear", C=0.025)),
            OneVsRestClassifier(LogisticRegression()),
            OneVsRestClassifier(MLPClassifier(alpha=1)),
            OneVsRestClassifier(LinearDiscriminantAnalysis()),
            OneVsRestClassifier(QuadraticDiscriminantAnalysis()),
            OneVsRestClassifier(AdaBoostClassifier())]

    results = []

    # Write result
    reportPath = getReportPath(dataPath, runType, dataPart)
    dir = os.path.dirname(reportPath)
    if not os.path.exists(dir):
        os.makedirs(dir)
    # iterate over classifiers
    f = open(reportPath, 'w')
    for name, clf in zip(names, classifiers):
        f.write(name + ":\n")
        f.write("##############################################\n")
        results.append(benchmarkModel(clf, name, X_train, X_test, y_train, y_test, target_names, True, f, runType, dataPath, dataPart, needRebuild))
    f.close()

    # results = [[x[i] for x in results] for i in range(2)]

    classes = [0, 1]

    # fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) ) = plt.subplots(nrows=3, ncols=3)
    #
    # count = 1
    # for clf_descr, confusion_matrix in results:
    #     if count == 1:
    #         plotMatrix(ax1, confusion_matrix, classes, clf_descr)
    #     elif count == 2:
    #         plotMatrix(ax2, confusion_matrix, classes, clf_descr)

    for clf_descr, confusion_matrix, title in results:
        fig = plt.figure(1)
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)
        thresh = confusion_matrix.max() / 2.
        plt.title(clf_descr)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, confusion_matrix[i, j],
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        savePath = getImagePath(dataPath, runType, dataPart, title)
        saveDir = os.path.dirname(savePath)
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        plt.savefig(savePath)
        plt.close(fig)

    return results

def plotMatrix(results, dataPath):
    classes = [0, 1]
    for clf_descr, confusion_matrix, title in results:
        fig = plt.figure(1)
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)
        thresh = confusion_matrix.max() / 2.
        plt.title(dataPath + "/" + runType + ": " + title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, confusion_matrix[i, j],
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        savePath = getImagePath(dataPath, runType, dataPart, title)
        saveDir = os.path.dirname(savePath)
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        plt.savefig(savePath)
        plt.close(fig)

def benchmarkModel(clf, name, X_train, X_test, y_train, y_test, target_names, isCluster, f, runType, dataPath, dataPart, needRebuild):

    print('****************************************')
    print('----------------------------')
    print('Running algorithm: ' + name)

    missing_values = ''
    if dataPath == 'data/fu_v2.csv':
        missing_values = 'Raw'
    elif dataPath == 'data/data_avg.csv':
        missing_values = 'AVG Values'
    else:
        missing_values = 'LinearRegression'

    print('Missing Values: ' + missing_values)
    print('Unbalanced Data: ' + runType)
    print('For data part ' + str(dataPart))
    print('----------------------------')

    if needRebuild or loadModel(dataPath, runType, dataPart, name) is None:
        t0 = time()
        clf.fit(X_train, y_train)
        train_time = time() - t0
        print("Train time: %0.3fs " % train_time)
        f.write("Train time: %0.3fs \n" % train_time)
        saveModel(dataPath, runType, dataPart, name, clf)
    else:
        print("Load saved model...")
        clf = loadModel(dataPath, runType, dataPart, name)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("Test time:  %0.3fs" % test_time)
    f.write("Test time: %0.3fs \n" % test_time)

    if isCluster:
        # Because result of predict contain multi label so we need refactor this result
        # Label 1-11 -> True if test label == 1
        pred = refineDropLabel(pred, True)

    # if names == "Logistic Regression":
    #     print(list(zip(clf.coef_, target_names)))

    # Show accuracy_score when run test with test set
    score = metrics.accuracy_score(y_test, pred)
    print("Correctly Classified Instances:   %0.3f" % score)
    f.write("Correctly Classified Instances:   %0.3f \n" % score)

    # score = metrics.accuracy_score(y_test, pred)
    # print("Incorrectly Classified Instances:   %0.3f" % score)
    # f.write("Incorrectly Classified Instances:   %0.3f \n" % score)

    #Show precision_score when run test with test set
    # score = metrics.precision_score(y_test, pred)
    # print("Precision Score:   %0.3f" % score)
    # f.write("Precision Score:   %0.3f \n" % score)
    #
    # #Show recall_score when run test with test set
    # score = metrics.recall_score(y_test, pred)
    # print("Recall Score:   %0.3f" % score)
    # f.write("Recall Score:   %0.3f \n" % score)
    #
    # #Show f1_score when run test with test set
    # score = metrics.f1_score(y_test, pred)
    # print("F1 Score:   %0.3f" % score)
    # f.write("F1 Score:   %0.3f \n" % score)

    #Show classification report
    print("Classification report:")
    f.write("Classification report: \n")
    print(metrics.classification_report(y_test, pred))
    f.write(metrics.classification_report(y_test, pred) + "\n")

    #Show confusion matrix
    print("Confusion Matrix:")
    f.write("Confusion Matrix: \n")
    confusion_matrix = metrics.confusion_matrix(y_test, pred)
    # print(confusion_matrix)
    f.write("a        b   <-- classified as \n")
    print("a        b   <-- classified as")
    f.write(str(confusion_matrix[0][0]) + "\t" + str(confusion_matrix[0][1]) + "    |    a = 0 \n")
    print(str(confusion_matrix[0][0]) + "\t" + str(confusion_matrix[0][1]) + "    |    a = 0")
    f.write(str(confusion_matrix[1][0]) + "\t" + str(confusion_matrix[1][1]) + "    |    b = 1 \n")
    print(str(confusion_matrix[1][0]) + "\t" + str(confusion_matrix[1][1]) + "    |    b = 1")
    f.write("\n")

    print('****************************************')

    # clf_descr = str(clf).split('(')[0]
    return dataPath + ": " + runType + ": " + name, confusion_matrix, name

# Use for cluster to change dropout label from 0 to 1
# Use for predict cluster result to change label 1-8 to 0
def refineDropLabel(y, forPredict):
    y_new = np.zeros(shape=(len(y),))
    count = 0
    for i in y:
        if forPredict == True:
            if i != 0:
                y_new[count] = 0
            else:
                y_new[count] = 1
        else:
            y_new[count] = 0
        count+=1

    return y_new

def getDataPath(dataPath):
    path = ""
    if dataPath == Paths.fu_v2:
        path = Paths.raw_report
    elif dataPath == Paths.data_avg:
        path = Paths.avg_report
    else:
        path = Paths.linear_report
    return path

# Use to get report path
def getReportPath(dataPath, run_type, i):
    path = getDataPath(dataPath)
    return path + "/" + str(i) + "/" + run_type + ".txt"

def getImagePath(dataPath, run_type, i, method):
    path = getDataPath(dataPath)
    return path + "/" + str(i) + "/" + run_type + "/" + method + ".png"

def getModelPath(dataPath, run_type, i, method):
    path = ""
    if dataPath == Paths.fu_v2:
        path = Paths.raw_model
    elif dataPath == Paths.data_avg:
        path = Paths.avg_model
    else:
        path = Paths.linear_model
    return path + "/" + str(i) + "/" + run_type + "/" + method + ""

def loadModel(dataPath, run_type, i, method):
    modelPath = getModelPath(dataPath, run_type, i, method)
    if not os.path.exists(modelPath):
        return None

    return joblib.load(modelPath)

def saveModel(dataPath, run_type, i, method, clf):
    modelPath = getModelPath(dataPath, run_type, i, method)
    saveDir = os.path.dirname(modelPath)
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    joblib.dump(clf, modelPath)
