"""
This module implements elmo indexation for the corpus.
"""

import numpy as np
import pickle
import tensorflow as tf

from time import time

from elmo_helpers import tokenize, get_elmo_vectors, load_elmo_embeddings
from constants import *


tf.reset_default_graph()

elmo_path = 'elmo'


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


batcher, sentence_character_ids, elmo_sentence_input = load_elmo_embeddings(elmo_path)


def doc_to_vec_elmo(text):
    """
    creates a vector out of a document using fasttext model

    :param text: string to be transformed into vector
    :return: np.array, mean vector of word vectors in provided text
      """
    tokens = [tokenize(text)]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        vector = crop_vec(get_elmo_vectors(sess, tokens, batcher, sentence_character_ids,
                                           elmo_sentence_input)[0], tokens[0])
    return vector


def elmo_indexing(tokenized):
    """
    indexing texts using ELMo
    :param tokenized: list of lists of str, tokenized documents from the corpus

    :return: matrix of document vectors
    """
    with tf.Session() as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.global_variables_initializer())
        indexed = []
        for i in range(200, len(tokenized)+1, 200):
            sentences = tokenized[i-200: i]
            elmo_vectors = get_elmo_vectors(
                sess, sentences, batcher, sentence_character_ids, elmo_sentence_input)

            for vect, sent in zip(elmo_vectors, sentences):
                cropped_vector = crop_vec(vect, sent)
                indexed.append(cropped_vector)
    return indexed


# elmo_doc2vec_corpus = elmo_indexing([tokenize(q) for q in data_lemm.texts[:2000]])
# with open("2000_raw_elmo_matrix.pickle", 'wb') as f:
#     pickle.dump(elmo_doc2vec_corpus, f)

with open("2000_raw_elmo_matrix.pickle", 'rb') as f:
    elmo_doc2vec_corpus = pickle.load(f)

def search_elmo(text, n=10):
    """
    searches a given query in elmo model, returns top n results

    :param text: str, text to search for
    :param n: int, how many most relevant results to return, 10 by default
    :return: list of tuples (float, str), cos_sim to the text, document text
    """
    query = " ".join(lemmatize(preprocess(text)))
    vector = doc_to_vec_elmo(query)
    cos_sim_relevance = [cos_sim(vector, doc_vec) for doc_vec in elmo_doc2vec_corpus]
    relevance_sorted_document_ids_top_n = enum_sort_tuple(cos_sim_relevance)[:n]
    return [(metric, data_lemm.texts[:2000][index]) for index, metric in relevance_sorted_document_ids_top_n]


print(search_elmo("нет меня печальнее на свете"))
