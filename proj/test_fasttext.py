import numpy as np
import pickle
import pandas as pd

from gensim.models.keyedvectors import KeyedVectors
from sklearn.preprocessing import normalize

from constants import *

model = KeyedVectors.load(path_fasttext_model)
questions = pd.read_csv("quora_question_pairs_rus.csv", index_col=0).dropna()
train_texts = questions[:2000]['question2'].tolist()


def doc_to_vec(lemmas):
    """
    creates a vector out of a document using fasttext model

    :param lemmas: list of str, lemmas to be transformed into vector
    :return: np.array, mean vector of word vectors in provided text
    """
    lemmas_vectors = np.zeros((len(lemmas), model.vector_size))
    doc_vec = np.zeros((model.vector_size,))
    for idx, lemma in enumerate(lemmas):
        if lemma in model.vocab:  # word in vocab
            try:
                lemmas_vectors[idx] = model[lemma]
            except AttributeError as e:  # word in vocab but not in model
                print(e)
    if lemmas_vectors.shape[0] is not 0:  # if lemmas_vectors is empty
        doc_vec = np.mean(lemmas_vectors, axis=0)
        return doc_vec

fasttext_doc2vec_corpus = []
for doc in train_texts:
    fasttext_doc2vec_corpus.append(doc_to_vec(lemmatize(preprocess(doc))))
#
# with open("2000_primitive_lemmatized_normalized_fasttext_matrix.pickle", "wb") as f:
#     pickle.dump(normalize(np.array(fasttext_doc2vec_corpus)), f)

# with open("2000_primitive_lemmatized_normalized_fasttext_matrix.pickle", "rb") as f:
#     fasttext_doc2vec_corpus = pickle.load(f)

def search_fasttext(query, n=10):
    """
    searches a given query in fasttext model, returns top n results

    :param text: str, query to be searched
    :param n: int, number of most relevant documents to be shown, 10 by default
    :return: list of tuples (float, str), metric and document, relevance ordered
    """
    query_vec = doc_to_vec(lemmatize(preprocess(query)))
    cos_sim_relevance = [cos_sim(query_vec, document) for document in fasttext_doc2vec_corpus]
    relevance_sorted_document_ids_top_n = enum_sort_tuple(cos_sim_relevance)[:n]
    return [(metric, train_texts[index]) for index, metric in
            relevance_sorted_document_ids_top_n]


query = "жизнь трудна но к счастью коротка"
print(search_fasttext(query))
