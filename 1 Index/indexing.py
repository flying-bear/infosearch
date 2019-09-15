##    Задача:
##    Data: Коллекция субтитров сезонов Друзьей. Одна серия - один документ.
##    To do: Постройте небольшой модуль поискового движка, который сможет осуществлять поиск по коллекции документов. На входе запрос и проиндексированная коллекция (в том виде, как посчитаете нужным), на выходе отсортированный по релевантности с запросом список документов коллекции.
##
##    Релизуйте:
##        - функцию препроцессинга данных
##            + unpack
##            + strip of punctuation
##            + to lower
##            + split into a bag of words
##            - lemmatize
##        - функцию индексирования данных
##            - raw frequency in term-document matrix
##            - TF-IDF (sklearn tfidfvectorizer)
##                - play with parametrs
##        - функцию метрики релевантности 
##        - собственно, функцию поиска

##    С помощью обратного индекса посчитайте:
##    a) какое слово является самым частотным
##    b) какое самым редким
##    c) какой набор слов есть во всех документах коллекции
##    d) какой сезон был самым популярным у Чендлера? у Моники?
##    e) кто из главных героев статистически самый популярный?

## TODO:
##    - read from zip
##    - reusable
##    - pep8

import numpy as np
import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

vectorizer = CountVectorizer(input = 'filename', encoding='utf-8')


def paths_from_dir(directory):
    paths = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if os.path.splitext(name)[1] == '.txt':
                paths.append(os.path.join(root, name))
    return paths


def fit_count_matrix(paths): # принимает список путей к файлам возвращает матрицу частотности (строки - документы, столбцы - слова)
    X = vectorizer.fit_transform(paths)
    matrix = X.toarray()
    return matrix


def tfidf_transform(count_matrix):
    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(count_matrix)
    return tfidf_matrix


def get_freq_list(count_matrix):
    return np.sum(count_matrix, axis=0)


def get_word_frequency_dictionary(freq_list, vocabulary):
    freq_dict = {}
    for i, word in enumerate(vocabulary):
        freq_dict[word] = word_freq[i]
    return freq_dict


def get_n_most_frequent(freq_list, vocabulary, n=10):
    indices = np.argpartition(freq_list, -n)[-n:]
    most_freq_dict = dict(zip(vocabulary[indices], freq_list[indices]))
    return most_freq_dict


def get_n_least_frequent(freq_list, vocabulary, n=1):
    indices = np.argpartition(freq_list, n)[n:]
    least_freq_dict = dict(zip(vocabulary[indices], freq_list[indices]))
    return least_freq_dict


def get_reverse_indexation(matrix):
    list_of_words = vectorizer.vocabulary_
    by_id = np.apply_along_axis(lambda col: [x[0] for x in sorted(enumerate(col), key=lambda x:x[1], reverse=True)], 0, matrix) 
    return dict(zip(list_of_words, by_id))


def main():
    loc = os.path.join(os.getcwd(),'friends')
    paths = paths_from_dir(loc)
    count_matrix = fit_count_matrix(paths)
    tfidf_matrix = tfidf_transform(count_matrix)
    vocabulary = np.array(vectorizer.get_feature_names())
    freq_list = get_freq_list(count_matrix)
    print(list(get_reverse_indexation(count_matrix).items())[:10])
##    a) какое слово является самым частотным
    most_frequent_word = get_n_most_frequent(freq_list, vocabulary, n=1)
##    b) какое самым редким
    least_frequent_words = get_n_least_frequent(freq_list, vocabulary)
##    c) какой набор слов есть во всех документах коллекции
    in_all_texts_bool = np.apply_along_axis(lambda x: 1 if not 0 in x else 0, 0, count_matrix)
    in_all_texts = vocabulary[np.argmax(in_all_texts_bool)]
##    d) какой сезон был самым популярным у Чендлера? у Моники?
##    e) кто из главных героев статистически самый популярный?


if __name__ == '__main__':
          main()
