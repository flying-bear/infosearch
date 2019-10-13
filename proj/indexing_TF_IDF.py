"""
This module creates TF-IDF indexation for the corpus.
"""

import numpy as np
import os
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from constants import enum_sort, cos_sim, morph, lemmatize, preprocess, count_vectorizer, trained_size

def corpus_to_tf_idf_matrix(list_of_texts, count_vectorizer=CountVectorizer(input = 'content', encoding='utf-8'),
                            save=True, target_location="tf_idf_matrix.npy"):
    """
    transforms a corpust into a tf-idf matrix

    :param list_of_texts: a list of preprocessed strings to be transformed
    :param count_vectorizer: sklearn CountVectorizer to be trained for the matrix, if not given - initialized by default
    :param save: bool, if True the matrix is saved, True by default
    :param target_location: where to save npy file with resulting, by default "tf_idf_matrix.npy"

    :return: np.ndarray, tf-idf matrix
    """
    transformer = TfidfTransformer()
    X = count_vectorizer.fit_transform(list_of_texts)
    count_matrix = X.toarray()
    tfidf_matrix = transformer.fit_transform(count_matrix).toarray()
    if save:
        np.save(target_location, tfidf_matrix)
    return tfidf_matrix

def search_tfidf(text, matrix, count_vectorizer):
    """
    searches a document in corpus by tf-idf

    :param text: string, query to be searched
    :param matrix: np.ndarray, tf-idf matrix
    :param count_vectorizer: sklearn CountVectorizer trained for the matrix

    :return: list of integers, relevance ordered ids of documents
    """
    vector = count_vectorizer.transform([text])
    cos_sim_list = np.apply_along_axis(cos_sim, 1, matrix, vector)
    return enum_sort(cos_sim_list)


def main():
    questions = pd.read_csv("quora_question_pairs_rus.csv", index_col=0).dropna()
    train_texts = questions[:trained_size]['question2'].tolist()
    preprocessed = [" ".join(preprocess(sent, lemm=True)) for sent in train_texts]
    arr1 = corpus_to_tf_idf_matrix(preprocessed, target_location="lemmatized_tf_idf_matrix.npy")
    arr = np.load(os.path.join(os.path.abspath("."), "lemmatized_tf_idf_matrix.npy"), allow_pickle=True)

if __name__ == "__main__":
    main()
