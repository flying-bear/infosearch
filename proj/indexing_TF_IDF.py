"""
This module implements TF-IDF indexation for the corpus.
"""

import numpy as np
import pickle

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
from time import time

from constants import *


class SearhTfidf:
    """
    indexes and searches a query by TF-IDF
    """
    def __init__(self, path_tfidf_matrix="", data=data_lemm):
        """
        computes or loads indexed tf-idf matrix for search
        :param path_tfidf_matrix: string, path to a pickle file with tf-idf matrix (loaded from file if given)
        :param data: lemmatized version of DataSet from constants, data_lemm by default
        """
        self.data = data
        if path_tfidf_matrix:
            self.matrix = self.load(path_tfidf_matrix)
        else:
            self.matrix = self.index()

    @staticmethod
    def load(path_tfidf_matrix):
        """
        loads data form a given path

        :param path_tfidf_matrix: string, path to a pickle file with tf-idf matrix
        :return: np.ndarray, tf-idf matrix from file
        """
        with open(path_tfidf_matrix, 'rb') as f:
            matrix = pickle.load(f)
        return matrix

    def index(self, path="lemmatized_normalized_tf_idf_matrix.pickle"):
        """
        computes and saves a normalized tf-idf matrix to path

        :param path: string, pickle file for matrix to be saved to,
            "lemmatized_normalized_tf_idf_matrix.pickle" by default
        :return: np.ndarray, tf-idf matrix from file
        """
        transformer = TfidfTransformer()
        tfidf_matrix = transformer.fit_transform(self.data.count_matrix).toarray()
        matrix = normalize(tfidf_matrix)
        with open(path, 'wb') as f:
            pickle.dump(matrix, f)
        return matrix

    def search(self, text, n=10):
        """
        searches a document in corpus by tf-idf

        :param text: string, query to be searched
        :param n: int, number of most relevant documents to be shown, 10 by default
        :return: list of integers, relevance ordered ids of documents
        """
        query = " ".join(lemmatize(preprocess(text)))
        vector = self.data.count_vectorizer.transform([query]).toarray()
        norm_vector = normalize(vector).reshape(-1, 1)
        cos_sim_list = np.dot(self.matrix, norm_vector)
        return [(metric[0], self.data.texts[index]) for index, metric in enum_sort_tuple(cos_sim_list)[:n]]



def main():
    #SearhTfidf()  # save the precomputed matrix
    start = time()
    tfidf = SearhTfidf("lemmatized_normalized_tf_idf_matrix.pickle")
    print(f"loading took {time()-start} sec")
    start = time()
    print(tfidf.search("горе горькое моё горе луковое"))
    print(f"searching took {time()-start} sec")
    start = time()
    print(tfidf.search("всё здесь мы сами себе ад"))
    print(f"searching took {time() - start} sec")


if __name__ == "__main__":
    main()
