# -*- coding: utf-8 -*-
"""
@title: Packetized Machine Learning in Support Vector Machines
@author: Ignacio Melero Miguel
"""
# Necessary imports.

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier


def create_training_testing_dataset(n_samples=10000, n_features=10, n_informative=10, n_redundant=0, n_repeated=0,
                                    n_classes=2, n_clusters_per_class=2, class_sep=1, flip_y=0.01, test_size=.3):

    # Generating the scenario.

    X, Y = datasets.make_classification(n_samples=int(n_samples), n_features=int(n_features),
                                        n_informative=int(n_informative), n_classes=int(n_classes),
                                        n_clusters_per_class=int(n_clusters_per_class),
                                        class_sep=int(class_sep), n_redundant=int(n_redundant),
                                        n_repeated=int(n_repeated), flip_y=flip_y)

    # Split data into training/testing sets. Training 70% and test 30% by default.

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

    # Standardize features by removing the mean and scaling to unit variance.

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, Y_train, Y_test


def create_only_training_dataset(n_samples=10000, n_features=10, n_informative=10, n_redundant=0, n_repeated=0,
                                 n_classes=2, n_clusters_per_class=2, class_sep=1, flip_y=0.01):

    # Generating the scenario.

    X_train, Y_train = datasets.make_classification(n_samples=int(n_samples), n_features=int(n_features),
                                                    n_informative=int(n_informative), n_classes=int(n_classes),
                                                    n_clusters_per_class=int(n_clusters_per_class),
                                                    class_sep=int(class_sep), n_redundant=int(n_redundant),
                                                    n_repeated=int(n_repeated), flip_y=flip_y)

    # Standardize features by removing the mean and scaling to unit variance.

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    return X_train, Y_train


def train(X_train, Y_train, gridsearch=False, kernel='linear'):

    if gridsearch is False:

        clf = svm.SVC(kernel=kernel)
        clf.fit(X_train, Y_train)

    else:

        clf_grid = svm.SVC(kernel=kernel)
        rang_C = np.logspace(0, 4, 4)
        tuned_params = [{'C': rang_C}]
        clf = GridSearchCV(clf_grid, tuned_params, cv=5).fit(X_train, Y_train)

    return clf


def train_packetized(X_train, Y_train, packets=10, gridsearch=False, kernel='linear'):

    X_train_packetized = []
    Y_train_packetized = []

    n = len(X_train)
    kf = KFold(n_splits=packets)

    for idx in kf.split(X_train):
        X_train_packetized.append(X_train[idx[1]])
        Y_train_packetized.append(Y_train[idx[1]])

    models = []

    for i in range(packets):

        if gridsearch is False:

            clf_packetized = svm.SVC(kernel=kernel)
            models.append(clf_packetized.fit(X_train_packetized[i], Y_train_packetized[i]))

        else:

            clf = svm.SVC(kernel=kernel)
            rang_C = np.logspace(0, 4, 4)
            tuned_params = [{'C': rang_C}]
            models.append(GridSearchCV(clf, tuned_params, cv=5).fit(X_train_packetized[i], Y_train_packetized[i]))

    return models


def train_bagging(X_train, Y_train):

    bdt = BaggingClassifier(SVC(kernel='rbf'), max_samples=0.1)
    bdt.fit(X_train, Y_train)

    return bdt


def predict(clf, X_test):

    Y_pred = clf.predict(X_test)

    return Y_pred


def most_common(lst):

    return max(set(lst), key=lst.count)


def predict_packetized(models, X_test):

    packets = len(models)

    Y_packetized_predict = []

    for i in range(packets):
        Y_packetized_predict.append(models[i].predict(X_test))

    Y_packetized_predict_pondered = []

    for i in range(len(Y_packetized_predict[0])):
        list_tmp = []
        for j in range(packets):
            list_tmp.append(Y_packetized_predict[j][i])
        Y_packetized_predict_pondered.append(most_common(list_tmp))

    return Y_packetized_predict_pondered


# ATTENTION: In order to use the function merged_model you need to modify the sklearn library, please
# contact imelero@tsc.uc3m.es to obtain the modified library.


def merged_model(X_train, models, packets):

    final_weights = []
    support_vectors = []
    n_support_0 = 0
    n_support_1 = 0
    support_index = []
    support_index_0 = []
    support_index_1 = []
    probA = []
    probB = []
    list_final_weights = []

    X_train_packetized = []
    kf = KFold(n_splits=packets)

    for idx in kf.split(X_train):
        X_train_packetized.append(X_train[idx[1]])

    for i in range(packets):
        n_support_0 = n_support_0 + models[i].n_support_[0]
        n_support_1 = n_support_1 + models[i].n_support_[1]
        support_index_0.extend(models[i].support_[:models[i].n_support_[0]] + len(X_train[idx[1]]) * i)
        support_index_1.extend(models[i].support_[(models[i].n_support_[0]):] + len(X_train[idx[1]]) * i)
        final_weights.extend(models[i]._dual_coef_[0])
        support_vectors.extend(models[i].support_vectors_)

    support_index.extend(support_index_0)
    support_index.extend(support_index_1)

    n_support_0_dtype32 = n_support_0.astype(np.int32)
    n_support_1_dtype32 = n_support_1.astype(np.int32)
    n_support_0_1 = [n_support_0_dtype32, n_support_1_dtype32]

    merge_model = svm.SVC(kernel='linear')
    merge_model.support_vectors_ = np.asarray(support_vectors)
    list_final_weights.append(np.asarray(final_weights))
    merge_model.dual_coef_ = np.asarray(list_final_weights)
    merge_model._dual_coef_ = np.asarray(list_final_weights)
    merge_model._intercept_ = models[0]._intercept_
    merge_model.gamma = 'auto'
    merge_model._gamma = 0.01
    merge_model.classes_ = np.asarray([0, 1])
    merge_model.probA_ = np.asarray(probA)
    merge_model.probB_ = np.asarray(probB)
    merge_model.n_support_ = np.asarray(n_support_0_1)
    merge_model._get_coef()
    # merge_model.fit_status_=0
    merge_model._sparse = False
    merge_model.support_ = np.asarray(support_index)

    return merge_model


# def plot(xlabel, ylabel, title, x1, y1, labelx1, x2, y2, labelx2, legend):
#
#     plt.figure()
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.plot(x1, y1, label=labelx1)
#     plt.plot(x2, y2, label=labelx2)
#     plt.legend(legend)
#     plt.show()