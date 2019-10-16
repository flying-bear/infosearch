import os


trained_size = 10000  # constant that defines further size of corpus for models to be trained on

path_fasttext_model = os.path.join("fasttext", "model.model")

path_tfidf_matrix = "lemmatized_normalized_tf_idf_matrix.pickle"
path_bm25_matrix = "lemmatized_normalized_bm25_matrix.pickle"
path_fasttext_matrix = "lemmatized_fasttext_matrix.pickle"
path_elmo_matrix = "elmo_matrix.pickle"