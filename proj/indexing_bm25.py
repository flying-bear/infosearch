"""
This module implements bm25 indexation for the corpus.
"""

import numpy as np
import pickle

from math import log
from sklearn.preprocessing import normalize
#from time import time

from constants import *


class SearchBM25:
    """
    indexes and searches a query by BM25
    """
    def __init__(self, path_bm25_matrix="", b=0.75, k=2.0):
        """
        computes or loads indexed tf-idf matrix for search

        :param path_bm25_matrix: string, path to a pickle file with tf-idf matrix (loaded from file if given)
        :param b: float, b coefficient in bm25 formula, 0.75 by default
        :param k: float, bm25 coefficient, 2.0 by default
        """
        self.b = b
        self.k = k
        self.N = trained_size  # collection size, imported from constants
        self.count_vectorizer = data_lemm.count_vectorizer
        if path_bm25_matrix:
            self.matrix = self.load(path_bm25_matrix)
        else:
            self.matrix = self.index()

    @staticmethod
    def load(path_bm25_matrix):
        """
        loads data form global data_lemm and given path

        :param path_bm25_matrix: string, path to a pickle file with tf-idf matrix
        :return: np.ndarray, tf-idf matrix from file
        """
        with open(path_bm25_matrix, 'rb') as f:
            return pickle.load(f)

    def compute_modified_idf(self, word, vocabulary, in_n_docs):
        """
        computes IDF modified for bm25

        :param word: string, word form vocabulary, IDF of which is to be computed
        :param vocabulary: list of strings (words)
        :param in_n_docs: list of ints corresponding to the number of documents a word occurred in
        :return: float, modified word IDF
        """
        word_id = vocabulary.index(word)
        n = in_n_docs[word_id]
        return log((self.N - n + 0.5) / (n + 0.5))

    def modify_tf(self, tf_value, doc_index, lens, avgdl):
        """
        modifies precomputed tf for bm25 formula

        :param tf_value: float, precomputed tf value
        :param doc_index: int, index of the document at hand
        :param lens: list of ints, document lengths
        :param avgdl: float, average document length
        :return: float, modified tf value
        """
        length = lens[doc_index]
        return (tf_value * (self.k + 1.0)) / (tf_value + self.k * (1.0 - self.b + self.b * (length / avgdl)))

    def compute_bm25(self, tf_matrix, lens, idfs, avgdl):
        """
        computes bm25 matrix

        :param tf_matrix: np.ndarray, tf matrix (count matrix divided by doc lengths)
        :param lens: list of ints, document lengths
        :param idfs: list of floats, word idfs
        :param avgdl: float, average document length
        :return: np.ndarray, bm25 matrix
        """
        enumed = np.ndenumerate(tf_matrix)
        for i, tf_value in enumed:
            doc_index = i[0]
            tf_matrix[i] = self.modify_tf(tf_value, doc_index, lens, avgdl)
        return tf_matrix * idfs

    def index(self, path="lemmatized_normalized_bm25_matrix.pickle"):
        """
        computes and saves a normalized tf-idf matrix to path

        :param path: string, pickle file for matrix to be saved to,
            "lemmatized_normalized_bm25_matrix.pickle" by default
        :return: np.ndarray, normalized tf-idf matrix
        """
        lens = [len(text.split()) for text in data_lemm.texts]
        avgdl = sum(lens) / self.N
        tf_matrix = data_lemm.count_matrix / np.array(lens).reshape((-1, 1))
        vocabulary = self.count_vectorizer.get_feature_names()
        in_n_docs = np.count_nonzero(data_lemm.count_matrix, axis=0)
        idfs = [self.compute_modified_idf(word, vocabulary, in_n_docs) for word in vocabulary]
        bm25_matrix = self.compute_bm25(tf_matrix, lens, idfs, avgdl)
        matrix = normalize(bm25_matrix)

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
        binary_vector = np.vectorize(lambda x: 1.0 if x != 0.0 else 0.0)(vector)
        norm_vector = normalize(binary_vector).reshape(-1, 1)
        cos_sim_list = np.dot(self.matrix, norm_vector)
        return [tup + (data_lemm.texts[tup[0]],) for tup in enum_sort_tuple([el[0] for el in cos_sim_list])[:n]]


def main():
    # start = time()
    # bm25 = SearchBM25("lemmatized_normalized_bm25_matrix.pickle")
    # print(f"loading took {time()-start} sec")
    # start = time()
    # print(bm25.search("Я дурак в воде, меня нет нигде..."))
    # print(f"searching took {time()-start} sec")
    # start = time()
    # print(bm25.search("а возлюбленный мой развернулся и ушёл"))
    # print(f"searching took {time() - start} sec")
    SearchBM25()  # save the precomputed matrix


if __name__ == "__main__":
    main()