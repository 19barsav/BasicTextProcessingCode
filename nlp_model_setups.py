
import numpy as np


from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import homogeneity_score, f1_score, accuracy_score, \
    rand_score, silhouette_score, v_measure_score, completeness_score

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import CategoricalNB

from sklearn import svm




def kmeans(X, true_k, max_iter, n_init, X_test, y_test):
    """
    kmeans model example
    made for vectorized text
    to create X and y, use

    :param X: training text data, parent member
    :param y: training true labels/classifications
            (not necessary for current fitness functions)
    :param true_k: number of clusters to make
    :param max_iter: max iterations, parent member
    :param n_init: n-initializations, parent member
    :param X_test: testing text data
    :param y_test: testing true labels/classifications
    :return: returns double - chosen metric
    """

    model = KMeans(n_clusters=true_k, init='k-means++', algorithm='lloyd', max_iter=max_iter, n_init=n_init)
    model.fit(np.asarray(X))
    predicted_labels = model.predict(X_test)

    return adjusted_rand_score(y_test, predicted_labels)



def svm_e(X, y,  X_test, y_test):
    """
    svm model example, with f1 score metric return

    :param X: training text data, parent member
    :param y: training true labels/classifications
            (not necessary for current fitness functions)
    :param X_test: testing text data
    :param y_test: testing true labels/classifications
    :return: returns double - f1-score
    """

    classifier_linear = svm.SVC(kernel='linear')
    classifier_linear.fit(np.asarray(X), y)
    predicted_labels = classifier_linear.predict(np.asarray(X_test.todense()))
    return f1_score(y_test, predicted_labels, average="weighted") + accuracy_score(y_test, predicted_labels)




def GNB(X, y, X_test, y_test):
    """
    Gaussian Naive Bayes model example, with f1 score metric return

    :param X: training text data, parent member
    :param y: training true labels/classifications
    :param X_test: testing text data
    :param y_test: testing true labels/classifications
    :return: returns double - f1-score
    """

    model = GaussianNB()
    model.fit(np.asarray(X), y)
    predicted_labels = model.predict(np.asarray(X_test.todense()))
    return f1_score(y_test, predicted_labels, average="weighted") + accuracy_score(y_test, predicted_labels)

