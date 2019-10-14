"""
This module implements TF-IDF indexation for the corpus.
"""

import numpy as np
import pickle

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer

from constants import *


class SearhTfidf:
    """
    indexes and searches a query by TF-IDF
    """
    def __init__(self, path_tfidf_matrix=""):
        """
        computes or loads indexed tf-idf matrix for search

        :param path_tfidf_matrix: string, path to a pickle file with tf-idf matrix (loaded from file if given)
        """
        self.count_vectorizer = data_lemm.count_vectorizer
        if path_tfidf_matrix:
            self.matrix = self.load(path_tfidf_matrix)
        else:
            self.matrix = self.index()

    @staticmethod
    def load(path_tfidf_matrix):
        """
        loads data form global data_lemm and given path

        :param path_tfidf_matrix: string, path to a pickle file with tf-idf matrix
        :return: np.ndarray, tf-idf matrix from file
        """
        with open(path_tfidf_matrix, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def index(path="lemmatized_normalized_tf_idf_matrix.pickle"):
        """
        computes and saves a normalized tf-idf matrix to path

        :param path: string, pickle file for matrix to be saved to,
            "lemmatized_normalized_tf_idf_matrix.pickle" by default
        :return: np.ndarray, normalized tf-idf matrix
        """
        transformer = TfidfTransformer()
        tfidf_matrix = transformer.fit_transform(data_lemm.count_matrix).toarray()
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
        if n >= trained_size:
            n = trained_size - 1
        vector = self.count_vectorizer.transform([text]).toarray()
        norm_vector = normalize(vector).reshape(-1, 1)
        cos_sim_list = np.dot(self.matrix, norm_vector)
        return [tup + (data_lemm.texts[tup[0]],) for tup in enum_sort_tuple([el[0] for el in cos_sim_list])[:n]]


def main():
    # start = time()
    # tfidf = SearhTfidf("lemmatized_normalized_tf_idf_matrix.pickle")
    # print(f"loading took {time()-start} sec")
    # start = time()
    # print(tfidf.search("Я дурак в воде, меня нет нигде..."))
    # print(f"searching took {time()-start} sec")
    # start = time()
    # print(tfidf.search("а возлюбленный мой развернулся и ушёл"))
    # print(f"searching took {time() - start} sec")
    SearhTfidf()  # save the precomputed matrix


if __name__ == "__main__":
    main()
