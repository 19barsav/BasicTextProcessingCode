from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

"""
Finds the numbers of clusters you should use when 
dong k-means. Graphs inertia and clusters. 
Find the elbow and you have your answers.
Takes roughly three decades to run. 
"""
def k_means():
    #CHANGE THIS FILE
    tweets_df2 = pd.read_csv('instrument_clean.csv')
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(tweets_df2['text_freq'])
    wcss = []

    for i in range(1, 7):
        kmeans = KMeans(n_clusters=i, verbose=True)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)


    plt.plot(range(1, 7), wcss)
    plt.title('Elbow Method for Aus')
    plt.xlabel('# of clusters')
    plt.ylabel('inertia')
    plt.show()
    #CHANGE ME
    plt.savefig("clust_elbow.jpg")



k_means()
