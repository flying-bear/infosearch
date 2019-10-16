"""
This module implements elmo indexation for the corpus.
"""

import numpy as np
import pickle
import tensorflow as tf

from time import time

from elmo_helpers import tokenize, get_elmo_vectors, load_elmo_embeddings
from constants import *


class SearchELMo:
    """
    indexes and searches a query by ELMo
    """
    def __init__(self, path_elmo_matrix="", data=data_lemm, elmo_path="elmo"):
        """
        computes or loads indexed tf-idf matrix for search
        :param path_elmo_matrix: string, path to a pickle file with elmo matrix (loaded from file if given)
        :param data: lemmatized version of DataSet from constants, data_lemm by default
        :param elmo_path: str, path to a folder with elmo model
        """
        self.data = data
        self.batcher, self.sentence_character_ids, self.elmo_sentence_input = load_elmo_embeddings(elmo_path)
        if path_elmo_matrix:
            self.matrix = self.load(path_elmo_matrix)
        else:
            self.matrix = self.index()

    @staticmethod
    def load(path_elmo_matrix):
        """
        loads data form a given path

        :param path_elmo_matrix: string, path to a pickle file with elmo matrix
        :return: np.ndarray, elmo matrix from file
        """
        logger.info(f"loading elmo from {path_elmo_matrix}")
        with open(path_elmo_matrix, 'rb') as f:
            matrix = pickle.load(f)
        return matrix

    @staticmethod
    def crop_vec(vect, sent):
        """
        deletes empty vectors at the end of a sentence

        :param vect: np.array, vector from ELMo
        :param sent: list of str, tokenized sentence
        :return: np.array
        """
        cropped_vector = vect[:len(sent), :]
        cropped_vector = np.mean(cropped_vector, axis=0)
        return cropped_vector

    def doc_to_vec(self, text):
        """
        creates a vector out of a document using fasttext model

        :param text: string to be transformed into vector
        :return: np.array, mean vector of word vectors in provided text
          """
        tokens = [tokenize(text)]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            vector = self.crop_vec(get_elmo_vectors(sess, tokens, self.batcher, self.sentence_character_ids,
                                                    self.elmo_sentence_input)[0], tokens[0])
        return vector

    def index(self, path="elmo_matrix.pickle"):
        """
        indexing texts using ELMo

        :return: matrix of document vectors
        """
        logger.info("indexing elmo matrix")
        tokenized = [tokenize(q) for q in self.data.texts]

        with tf.Session() as sess:
            # It is necessary to initialize variables once before running inference.
            sess.run(tf.global_variables_initializer())
            indexed = []
            for i in range(200, len(tokenized)+1, 200):
                sentences = tokenized[i-200: i]
                elmo_vectors = get_elmo_vectors(sess, sentences,
                                                self.batcher, self.sentence_character_ids, self.elmo_sentence_input)

                for vect, sent in zip(elmo_vectors, sentences):
                    cropped_vector = self.crop_vec(vect, sent)
                    indexed.append(cropped_vector)

        matrix = np.array(indexed)
        logger.info("saving elmo matrix")
        with open(path, 'wb') as f:
            pickle.dump(matrix, f)
        return matrix

    def search(self, text, n=10):
        """
        searches a given query in elmo model, returns top n results

        :param text: str, text to search for
        :param n: int, how many most relevant results to return, 10 by default
        :return: list of tuples (float, str), cos_sim to the text, document text
        """
        if n >= trained_size:
            n = trained_size - 1

        logger.info(f"searching for {n} most relevant results for '{text}' in elmo model")

        query = " ".join(preprocess(text))
        vector = self.doc_to_vec(query)
        cos_sim_relevance = [cos_sim(vector, doc_vec) for doc_vec in np.array(self.matrix)]
        relevance_sorted_document_ids_top_n = enum_sort_tuple(cos_sim_relevance)[:n]
        return [(metric, self.data.texts[index]) for index, metric in relevance_sorted_document_ids_top_n]


def main():
    # start = time()
    # SearchELMo()  # save the precomputed matrix
    # print(f"indexing took {time()-start} sec")
    start = time()
    elmo = SearchELMo("elmo_matrix.pickle")
    print(f"loading took {time()-start} sec")
    start = time()
    print(elmo.search("я так хочу спать нет сил"))
    print(f"searching took {time()-start} sec")
    start = time()
    print(elmo.search("интересно, заработет ли здесь элмо"))
    print(f"searching took {time() - start} sec")


if __name__ == "__main__":
    main()
