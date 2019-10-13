"""
This module manages the different searches.
"""

import numpy as np
import os
import pickle

from sklearn.preprocessing import normalize

from constants import enum_sort, cos_sim

def search_tfidf(text, matrix, count_vectorizer):
    """
    searches a document in corpus by tf-idf

    :param text: string, query to be searched
    :param matrix: np.ndarray, tf-idf matrix
    :param count_vectorizer: sklearn CountVectorizer trained for the matrix

    :return: list of integers, relevance ordered ids of documents
    """
    vector = count_vectorizer.transform([text]).toarray()
    norm_vector = normalize(vector).reshape(-1,1)
    cos_sim_list = np.dot(matrix, norm_vector)
    return enum_sort(cos_sim_list)


def main():
    with open (os.path.join(os.path.abspath("."), "lemmatized_tf_idf_matrix.pickle"), 'rb') as f:
        tf_idf_matrix = pickle.load(f)
    with open("lemmatized_count_vectorizer.pickle", "rb") as f:
        count_vectorizer = pickle.load(f)
    print(search_tfidf("я дурак в воде меня нет нигде", tf_idf_matrix, count_vectorizer)[:10])


if __name__ == "__main__":
    main()