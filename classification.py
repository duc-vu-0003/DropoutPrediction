from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
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

# We run classify ...
def classifiers(X_train, X_test, y_train, y_test, target_names, dataPath, runType, dataPart):
    names = ["K- Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Network", "AdaBoost",
         "Naive Bayes", "QDA", "Logistic Regression"]

    classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
            LogisticRegression()]

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
        results.append(benchmarkModel(clf, name, X_train, X_test, y_train, y_test, target_names, False, f))
    f.close()

    # # iterate over classifiers
    # for name, clf in zip(names, classifiers):
    #     print(name + ":")
    #     results.append(benchmarkModel(clf, name, X_train, X_test, y_train, y_test, target_names, False, dataPath, runType, dataPart))
    #     # score = clf.score(X_test, y_test)
    #     # duration = time() - t0
    #     # print("Done in %fs" % duration)
    #     # print()
    #
    # indices = np.arange(len(results))
    # results = [[x[i] for x in results] for i in range(4)]
    # clf_names, score, training_time, test_time = results
    # training_time = np.array(training_time) / np.max(training_time)
    # test_time = np.array(test_time) / np.max(test_time)

def classifiersMultiLabel(X_train, X_test, y_train, y_test, target_names, dataPath, runType, dataPart):
    names = ["K- Nearest Neighbors", #"Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Network", "AdaBoost",
         "Naive Bayes", "QDA", "Logistic Regression"]

    classifiers = [
            OneVsRestClassifier(KNeighborsClassifier(3)),
            # OneVsRestClassifier(SVC(kernel="linear", C=0.025)),
            # OneVsRestClassifier(SVC(gamma=2, C=1)),
            OneVsRestClassifier(DecisionTreeClassifier(max_depth=5)),
            OneVsRestClassifier(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
            OneVsRestClassifier(MLPClassifier(alpha=1)),
            OneVsRestClassifier(AdaBoostClassifier()),
            OneVsRestClassifier(GaussianNB()),
            OneVsRestClassifier(QuadraticDiscriminantAnalysis()),
            OneVsRestClassifier(LogisticRegression())]

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
        results.append(benchmarkModel(clf, name, X_train, X_test, y_train, y_test, target_names, True, f))
    f.close()

def benchmarkModel(clf, names, X_train, X_test, y_train, y_test, target_names, isCluster, f):
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("Train time: %0.3fs " % train_time)
    f.write("Train time: %0.3fs \n" % train_time)

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

    #Show precision_score when run test with test set
    score = metrics.precision_score(y_test, pred)
    print("Precision Score:   %0.3f" % score)
    f.write("Precision Score:   %0.3f \n" % score)
    #
    # #Show recall_score when run test with test set
    score = metrics.recall_score(y_test, pred)
    print("Recall Score:   %0.3f" % score)
    f.write("Recall Score:   %0.3f \n" % score)
    #
    # #Show f1_score when run test with test set
    score = metrics.f1_score(y_test, pred)
    print("F1 Score:   %0.3f" % score)
    f.write("F1 Score:   %0.3f \n" % score)

    #Show classification report
    print("Classification report:")
    f.write("Classification report: \n")
    print(metrics.classification_report(y_test, pred))
    f.write(metrics.classification_report(y_test, pred) + "\n")

    #Show confusion matrix
    print("Confusion Matrix:")
    f.write("Confusion Matrix: \n")
    confusion_matrix = metrics.confusion_matrix(y_test, pred)
    print(confusion_matrix)
    f.write("a    b   <-- classified as \n")
    f.write(str(confusion_matrix[0][0]) + "\t" + str(confusion_matrix[0][1]) + " |    a = 0 \n")
    f.write(str(confusion_matrix[1][0]) + "\t" + str(confusion_matrix[1][1]) + " |    a = 1 \n")
    f.write("\n")

    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

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

# Use to get report path
def getReportPath(dataPath, run_type, i):
    path = ""
    if dataPath == Paths.fu_v2:
        path = Paths.raw_report
    elif dataPath == Paths.data_avg:
        path = Paths.avg_report
    else:
        path = Paths.linear_report
    return path + "/" + str(i) + "/" + run_type + ".txt"
