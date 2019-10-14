"""
This module creates TF-IDF indexation for the corpus.
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
        if path_tfidf_matrix:
            self.load(path_tfidf_matrix)
        else:
            self.index()

    def load(self, path_tfidf_matrix,):
        """
        loads data form global data_lemm and given path

        :param path_tfidf_matrix: string, path to a pickle file with tf-idf matrix
        :return: self
        """
        self.count_vectorizer = data_lemm.count_vectorizer
        with open(path_tfidf_matrix, 'rb') as f:
            self.matrix = pickle.load(f)
        return self

    def index(self, path="lemmatized_normalized_tf_idf_matrix.pickle"):
        """
        computes and saves a normalized tf-idf matrix to path

        :param path: string, pickle file for matrix to be saved to,
            "lemmatized_normalized_tf_idf_matrix.pickle" by default
        :return: self
        """
        transformer = TfidfTransformer()
        tfidf_matrix = transformer.fit_transform(data_lemm.count_matrix).toarray()
        self.matrix = normalize(tfidf_matrix)
        with open(path, 'wb') as f:
            pickle.dump(self.matrix, f)
        return self

    def search(self, text):
        """
        searches a document in corpus by tf-idf

        :param text: string, query to be searched
        :return: list of integers, relevance ordered ids of documents
        """
        vector = self.count_vectorizer.transform([text]).toarray()
        norm_vector = normalize(vector).reshape(-1, 1)
        cos_sim_list = np.dot(self.matrix, norm_vector)
        return [tup + (data_lemm.texts[tup[0]],) for tup in enum_sort_tuple([el[0] for el in cos_sim_list])]

def main():
    # print(SearhTfidf("lemmatized_tf_idf_matrix.pickle").search("я дурак в воде меня нет нигде")[:10])
    SearhTfidf() # save the precomputed matrix

if __name__ == "__main__":
    main()
