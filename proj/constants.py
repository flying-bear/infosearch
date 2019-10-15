"""
This module contains constants and functions to be used in all other modules of the project.
"""

import numpy as np
import pandas as pd
import pickle
import pymorphy2
import re

from sklearn.feature_extraction.text import CountVectorizer

morph = pymorphy2.MorphAnalyzer()
trained_size = 10000  # constant that defines further size of corpus for models to be trained on


def lemmatize(list_of_words):
    """
    lemmatizes a list of strings using PyMorphy

    :param list_of_words: list of strings to be lemmatized
    :return: list of lemmatized word strings
    """
    # [morph.parse(word)[0].normal_form for word in list_of_words]
    return list(map(lambda word: morph.parse(word)[0].normal_form, list_of_words))



def preprocess(text):
    """
    cleans a text of punctuation and case

    :param text: string to be cleaned
    :return: a list of strings (words) in lowercase and stripped of punctuation
    """
    low = text.lower()
    stripped = re.sub("!|\.|,|#|$|%|\\|\'|\(|\)|-|\+|\*|/|\:|;|<|>|=|\?|\[|\]|@|^|_|`|{|}|~", "", low)
    words = stripped.split()
    return words


def enum_sort_tuple(arr):
    """
    sorts list by values and returns sorted ids

    :param arr: list to be sorted
    :return: list of tuples (int, float),  ids and values sorted by the values in arr
    """
    return sorted(enumerate(arr), key=lambda x: x[1], reverse=True)


def cos_sim(v1, v2):
    """
    cosine similarity of two vectors

    :param v1: np.array, first vector
    :param v2: np.array, second vector
    :return: float, cosine similarity of the vectors
    """
    return np.inner(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_data():
    """
    reads quora_question_pairs_rus dataset from file and applies preprocessing

    :return: a list of preprocessed strings-texts
    """
    questions = pd.read_csv("quora_question_pairs_rus.csv", index_col=0).dropna()
    train_texts = questions[:trained_size]['question2'].tolist()
    # [" ".join(preprocess(sent)) for sent in train_texts]
    preprocessed =  list(map(lambda sent: " ".join(preprocess(sent)), train_texts))
    return preprocessed


def get_counts(list_of_texts):
    """
    calculates a row count doc2term matrix (rows - documents, columns - words)

    :param list_of_texts: a list of strings (documents)
    :return count_matrix: np.ndarray, matrix of word counts for each document
    :return count_vectorizer: fitted CountVectorizer from sklearn
    """
    count_vectorizer = CountVectorizer(input='content', encoding='utf-8')
    X = count_vectorizer.fit_transform(list_of_texts)
    count_matrix = X.toarray()
    return count_matrix, count_vectorizer


class DataSet:
    def __init__(self, lemm=False):
        self.texts = get_data()
        if lemm:
            # [" ".join(lemmatize(sent.split())) for sent in self.texts]
            self.lemmatized_texts = list(map(lambda sent: " ".join(lemmatize(sent.split())), self.texts))
            self.count_matrix, self.count_vectorizer = get_counts(self.lemmatized_texts)
            path = "lemmatized_count_vectorizer.pickle"
        else:
            path = "raw_count_vectorizer.pickle"
            self.count_matrix, self.count_vectorizer = get_counts(self.texts)
        with open(path, 'wb') as f:
            pickle.dump(self.count_vectorizer, f)


data_lemm = DataSet(lemm=True)
data_raw = DataSet()
