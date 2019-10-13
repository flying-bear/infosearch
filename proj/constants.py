"""
This module contains constants and functions to be used in all other modules of the project.
"""

import numpy as np
import pandas as pd
import pymorphy2
import re

from sklearn.feature_extraction.text import CountVectorizer

morph=pymorphy2.MorphAnalyzer()
trained_size = 10000 # constant that defines further size of corpus for models to be trained on


def lemmatize(list_of_words):
    """
    lemmatizes a list of strings using PyMorphy

    :param list_of_words: list of strings to be lemmatized
    :return: list of lemmatized word strings
    """
    return [morph.parse(word)[0].normal_form for word in list_of_words]


def preprocess(text, lemm=False):
    """
    cleans a text of punctuation and case

    :param text: string to be cleaned
    :param lemm: bool, if the words should be lemmatized, False by default
    :return: a list of strings (words) in lowercase and stripped of punctuation
    """
    low = text.lower()
    stripped = re.sub("!|\.|,|#|$|%|\\|\'|\(|\)|-|\+|\*|/|\:|;|<|>|=|\?|\[|\]|@|^|_|`|{|}|~", "", low)
    words = stripped.split()
    if lemm:
        words = lemmatize(words)
    return words


def enum_sort(arr):
    """
    sorts list by values and returns sorted ids
    :param arr: list to be sorted
    :return: list of ids sorted by thir values in arr
    """
    return [x[0] for x in sorted(enumerate(arr), key=lambda x:x[1], reverse=True)]


def cos_sim(v1, v2):
    """
    cosine similarity of two vectors

    :param v1: np.array, first vector
    :param v2: np.array, second vector
    :return: float, cosine similarity of the vectors
    """
    return np.inner(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_data(lemm=False):
    """
    reads quora_question_pairs_rus dataset from file and applies preprocessing
    :param lemm: bool, if True lemmatiztion is applied, False by default
    :return: a list of preprocessed strings-texts
    """
    questions = pd.read_csv("quora_question_pairs_rus.csv", index_col=0).dropna()
    train_texts = questions[:trained_size]['question2'].tolist()
    preprocessed = [" ".join(preprocess(sent, lemm=lemm)) for sent in train_texts]
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