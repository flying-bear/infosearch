"""
This module creates TF-IDF indexation for the corpus.
"""

import numpy as np
import pickle

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
from constants import *


def main():
    lemm_train_texts = get_data(lemm=True)
    count_matrix, count_vectorizer = get_counts(lemm_train_texts)
    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(count_matrix).toarray()
    tfidf_matrix_norm = normalize(tfidf_matrix)
    with open("lemmatized_count_vectorizer.pickle", 'wb') as f:
        pickle.dump(count_vectorizer, f)
    with open ("lemmatized_tf_idf_matrix.pickle", 'wb') as f:
        pickle.dump(tfidf_matrix_norm, f)

if __name__ == "__main__":
    main()
