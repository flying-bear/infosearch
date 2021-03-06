"""
This module implements fasttext indexation for the corpus.
"""

import numpy as np
import pickle

from gensim.models.keyedvectors import KeyedVectors
from time import time

from utils import logger, data_lemm
from utils import preprocess, lemmatize, enum_sort_tuple, cos_sim

from config import path_fasttext_model


class SearchFastText:
    """
    indexes and searches a query by fasttext
    """
    def __init__(self, path_fasttext_matrix="", path_model=path_fasttext_model, data=data_lemm):
        """
        computes or loads indexed fasttext matrix for search

        :param path_model: str, path to a gensim fasttext model, imported from constants by default
        :param path_fasttext_matrix: str, path to a pickle file with fasttext matrix (loaded from file if given)
        :param data: lemmatized version of DataSet from constants, data_lemm by default
        """
        self.model = KeyedVectors.load(path_model)
        self.data = data
        if path_fasttext_matrix:
            self.matrix = self.load(path_fasttext_matrix)
        else:
            self.matrix = self.index()

    def load(self, path_fasttext_matrix):
        """
        loads data form a given path

        :param path_fasttext_matrix: str, path to a pickle file with fasttext matrix
        :return: np.ndarray, fasttext matrix from file
        """
        logger.info(f"loading fasttext from {path_fasttext_matrix}")
        try:
            with open(path_fasttext_matrix, 'rb') as f:
                matrix = pickle.load(f)
            return matrix
        except FileNotFoundError as ex:
            logger.critical(f"file not found: '{ex}'")
            return self.index()

    def doc_to_vec(self, lemmas):
        """
        creates a vector out of a document using fasttext model

        :param lemmas: list of str, lemmas to be transformed into vector
        :return: np.array, mean vector of word vectors in provided text
        """
        lemmas_vectors = np.zeros((len(lemmas), self.model.vector_size))
        doc_vec = np.zeros((self.model.vector_size,))
        for idx, lemma in enumerate(lemmas):
            if lemma in self.model.vocab:  # word in vocab
                try:
                    lemmas_vectors[idx] = self.model[lemma]
                except AttributeError as e:  # word in vocab but not in model
                    logger.warning(f"word {lemma} is in model vocab, but not in the model ")
                    logger.info(f"exception message: {e}")
        if lemmas_vectors.shape[0] is not 0:  # if lemmas_vectors is empty
            doc_vec = np.mean(lemmas_vectors, axis=0)
            return doc_vec

    def index(self, path="lemmatized_fasttext_matrix.pickle"):
        """
        computes and saves a fasttext matrix to path

        :param path: str, pickle file for matrix to be saved to,
            "lemmatized_fasttext_matrix.pickle" by default
        :return: np.ndarray, fasttext matrix
        """
        logger.info("indexing fasttext matrix")
        fasttext_doc2vec_matrix = []
        for doc in self.data.lemmatized_texts:
            fasttext_doc2vec_matrix.append(self.doc_to_vec(doc.split()))
        matrix = fasttext_doc2vec_matrix
        logger.info("saving fasttext matrix")
        with open(path, 'wb') as f:
            pickle.dump(matrix, f)
        return matrix

    def search(self, text, n=10):
        """
        searches a given query in fasttext model, returns top n results

        :param text: str, query to be searched
        :param n: int, number of most relevant documents to be shown, 10 by default
        :return: list of tuples (float, str), metric and document, relevance ordered
        """
        if n >= self.data.length:
            n = self.data.length - 1

        logger.info(f"searching for {n} most relevant results for '{text}' in fasttext model")

        vector = self.doc_to_vec(lemmatize(preprocess(text)))
        cos_sim_list = [cos_sim(vector, document) for document in self.matrix]
        return [(metric, self.data.texts[index]) for index, metric in enum_sort_tuple(cos_sim_list)[:n]]


def main():
    # SearchFastText()  # save the precomputed matrix
    start = time()
    ft = SearchFastText("lemmatized_fasttext_matrix.pickle")
    print(f"loading took {time()-start} sec")
    start = time()
    print(ft.search("я никогда не сдам программирование я очень глупая"))
    print(f"searching took {time()-start} sec")
    start = time()
    print(ft.search("жизнь трудна но к счастью коротка"))
    print(f"searching took {time() - start} sec")


if __name__ == "__main__":
    main()
