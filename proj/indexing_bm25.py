"""
This module implements bm25 indexation for the corpus.
"""

import numpy as np
import pickle

from math import log
from sklearn.preprocessing import normalize
from time import time

from utils import logger, data_lemm
from utils import preprocess, lemmatize, enum_sort_tuple


class SearchBM25:
    """
    indexes and searches a query by BM25
    """
    def __init__(self, path_bm25_matrix="", data=data_lemm, b=0.75, k=2.0):
        """
        computes or loads indexed bm25 matrix for search

        :param path_bm25_matrix: string, path to a pickle file with bm25 matrix (loaded from file if given)
        :param data: lemmatized version of DataSet from constants, data_lemm by default
        :param b: float, b coefficient in bm25 formula, 0.75 by default
        :param k: float, bm25 coefficient, 2.0 by default
        """
        self.data = data
        self.b = b
        self.k = k
        if path_bm25_matrix:
            self.matrix = self.load(path_bm25_matrix)
        else:
            self.matrix = self.index()

    def load(self, path_bm25_matrix):
        """
        loads data form a given path

        :param path_bm25_matrix: string, path to a pickle file with bm25 matrix
        :return: np.ndarray, bm25 matrix from file
        """
        logger.info(f"loading bm25 from {path_bm25_matrix}")
        try:
            with open(path_bm25_matrix, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError as ex:
            logger.critical(f"file not found: '{ex}'")
            return self.index()

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
        return log((self.data.length - n + 0.5) / (n + 0.5))

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
        computes and saves a normalized bm25matrix to path

        :param path: string, pickle file for matrix to be saved to,
            "lemmatized_normalized_bm25_matrix.pickle" by default
        :return: np.ndarray, normalized bm25 matrix
        """
        logger.info("indexing bm25 matrix")
        lens = [len(text.split()) for text in self.data.texts]
        avgdl = sum(lens) / self.data.length
        tf_matrix = self.data.count_matrix / np.array(lens).reshape((-1, 1))
        vocabulary = self.data.count_vectorizer.get_feature_names()
        in_n_docs = np.count_nonzero(self.data.count_matrix, axis=0)
        # [self.compute_modified_idf(word, vocabulary, in_n_docs) for word in vocabulary]
        idfs = map(lambda word: self.compute_modified_idf(word, vocabulary, in_n_docs), vocabulary)
        bm25_matrix = self.compute_bm25(tf_matrix, lens, idfs, avgdl)
        matrix = normalize(bm25_matrix)

        logger.info("saving bm25 matrix")
        with open(path, 'wb') as f:
            pickle.dump(matrix, f)
        return matrix

    def search(self, text, n=10):
        """
        searches a document in corpus by tf-idf

        :param text: string, query to be searched
        :param n: int, number of most relevant documents to be shown, 10 by default
        :return: list of tuples (float, str), metric and document, relevance ordered
        """
        if n >= self.data.length:
            n = self.data.length - 1

        logger.info(f"searching for {n} most relevant results for '{text}' in bm25 model")

        query = " ".join(lemmatize(preprocess(text)))
        vector = self.data.count_vectorizer.transform([query]).toarray()
        norm_vector = normalize(vector).reshape(-1, 1)
        cos_sim_list = np.dot(self.matrix, norm_vector)
        return [(metric[0], self.data.texts[index]) for index, metric in enum_sort_tuple(cos_sim_list)[:n]]


def main():
    # SearchBM25()  # save the precomputed matrix
    start = time()
    bm25 = SearchBM25("lemmatized_normalized_bm25_matrix.pickle")
    print(f"loading took {time()-start} sec")
    start = time()
    print(bm25.search("Я дурак в воде, меня нет нигде..."))
    print(f"searching took {time()-start} sec")
    start = time()
    print(bm25.search("а возлюбленный мой развернулся и ушёл"))
    print(f"searching took {time() - start} sec")


if __name__ == "__main__":
    main()