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
from time import time
from sklearn.utils.extmath import density
from sklearn import metrics
import numpy as np
from sklearn.multiclass import OneVsRestClassifier

# We run classify ...
def classifiers(X_train, X_test, y_train, y_test, target_names):
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
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        print(name + ":")
        results.append(benchmarkModel(clf, name, X_train, X_test, y_train, y_test, target_names, False))
        # score = clf.score(X_test, y_test)
        # duration = time() - t0
        # print("Done in %fs" % duration)
        # print()

    indices = np.arange(len(results))
    results = [[x[i] for x in results] for i in range(4)]
    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    # plt.figure(figsize=(12, 8))
    # plt.title("Score")
    # plt.barh(indices, score, .2, label="score", color='navy')
    # plt.barh(indices + .3, training_time, .2, label="training time",
    #      color='c')
    # plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
    # plt.yticks(())
    # plt.legend(loc='best')
    # plt.subplots_adjust(left=.25)
    # plt.subplots_adjust(top=.95)
    # plt.subplots_adjust(bottom=.05)
    #
    # for i, c in zip(indices, clf_names):
    #     plt.text(-.3, i, c)
    #
    # plt.show()

def classifiersMultiLabel(X_train, X_test, y_train, y_test, target_names):
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
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        print(name + ":")
        results.append(benchmarkModel(clf, name, X_train, X_test, y_train, y_test, target_names, True))
        # score = clf.score(X_test, y_test)
        # duration = time() - t0
        # print("Done in %fs" % duration)
        # print()

def benchmarkModel(clf, names, X_train, X_test, y_train, y_test, target_names, isCluster):
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    if isCluster:
        # Because result of predict contain multi label so we need refactor this result
        # Label 1-11 -> True if test label == 1
        pred = refineDropLabel(pred, True)

    # if names == "Logistic Regression":
    #     print(list(zip(clf.coef_, target_names)))

    #Show accuracy_score when run test with test set
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    #Show precision_score when run test with test set
    score = metrics.precision_score(y_test, pred)
    print("precision_score:   %0.3f" % score)
    #
    # #Show recall_score when run test with test set
    score = metrics.recall_score(y_test, pred)
    print("recall_score:   %0.3f" % score)
    #
    # #Show f1_score when run test with test set
    score = metrics.f1_score(y_test, pred)
    print("f1_score:   %0.3f" % score)

    #Show classification report
    print("classification report:")
    print(metrics.classification_report(y_test, pred))

    #Show confusion matrix
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

# Use for cluster to change dropout label from 1 to 0
# Use for predict cluster result to change label 1-8 to 1
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
