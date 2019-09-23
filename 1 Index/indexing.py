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
from pymystem3 import Mystem
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def enum_sort(arr): # takes a list and returns a list of ids in the decreasing order of the values from the input
    return [x[0] for x in sorted(enumerate(arr), key=lambda x:x[1], reverse=True)]


def text_strip_split(text): # takes a text and returns a list of words in lowercase and stripped of punctuation
    low = text.lower()
    stripped = re.sub('\.|,|#|$|%|\\|\'|\(|\)|-|\+|\*|/|\:|;|<|>|=|\?|\[|\]|@|^|_|`|{|}|~', '', low)
    words = stripped.split()
    return words


mystem_analyzer = Mystem()
def tokenize_and_lemmatize(text): # takes a text returns a list of lemmas
    words = re.findall('\w+', text)
    lemmas = [mystem_analyzer.lemmatize(word)[0] for word in words]
    return lemmas

vectorizer = CountVectorizer(input = 'filename', encoding='utf-8') #tokenizer=tokenize_and_lemmatize takes ages for some reason



def paths_from_dir(directory): # takes a directory returns a list of paths to all txt files
    paths = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if os.path.splitext(name)[1] == '.txt':
                paths.append(os.path.join(root, name))
    return paths


def fit_count_matrix(paths): # takes a list of paths returns a doc to term raw count matrix (rows - documents, columns - words)
    X = vectorizer.fit_transform(paths)
    matrix = X.toarray()
    return matrix


def tfidf_transform(count_matrix): # takes a count matrix returns a tf-idf doc to term matrix
    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(count_matrix)
    return tfidf_matrix


def get_freq_list(count_matrix): # takes a count matrix returns a list of frequencies for all the words
##                                  (word is coded by the number of column, or its position in the list)
    return np.sum(count_matrix, axis=0)


def get_word_frequency_dictionary(freq_list, vocabulary): # takes a frequecies list in order and a list of all words in order
##                                                           returns a dictionary (keys - words, values - frequency)
    freq_dict = {}
    for i, word in enumerate(vocabulary):
        freq_dict[word] = word_freq[i]
    return freq_dict


def get_n_most_frequent(freq_list, vocabulary, n=10): # takes a frequecies list in order and a list of all words in order,
##                                                      returns a dictionary of n most frequent words and their frequencies as dictionary values
    indices = np.argpartition(freq_list, -n)[-n:]
    most_freq_dict = dict(zip(vocabulary[indices], freq_list[indices]))
    return most_freq_dict


def get_n_least_frequent(freq_list, vocabulary, n=1): # takes a frequecies list in order and a list of all words in order,
##                                                          returns a dictionary of all the words with n least frequent ranks (one to n instances) 
##                                                          and their frequencies as dictionary values
    indices = np.argpartition(freq_list, n)[n:]
    least_freq_dict = dict(zip(vocabulary[indices], freq_list[indices]))
    return least_freq_dict


def get_reverse_indexation(matrix, vocabulary): # takes a doc to term matrix  and vocabulary returns a reverse indexation:
##                                      a dictionary (keys - words, values - an ordered list of document numbers)
    by_id = np.apply_along_axis(enum_sort, 0, matrix) 
    return dict(zip(vocabulary, by_id))


def get_meta(paths): # takes a list of paths returns a list with a tuple (season, episode) about each episode, where order is inhereted from path order
    meta = []
    for path in paths:
        name = os.path.split(path)[1].split('.')[0]
        ep_num, ep_name = name.split(' - ')[1:]
        ep_name = ep_name.strip('ru.txt')
        season, episode = ep_num.split('x')
        season = int(season)
        meta.append((season, episode))
    return meta


def search(text, matrix, vocabulary): # takes a text and a doc to term matrix and its vocabulary returns a relevance-ordered list of document ids 
    vector = vectorizer.transform([text]).reshape(1, -1)
    cos_sim_list = []
    for row in matrix:
        cos_sim_list.append(cosine_similarity(vector, row.reshape(1, -1))[0])
    return enum_sort(cos_sim_list)


def search_by_season(text, season, matrix, vocabulary, meta): # takes a text and the number of season and and a doc to term matrix and its vocabulary
##                                                                and a list of swithces in season numbers
##                                                                returns relevance-ordered list of document ids in the given season
    from_id = meta[season - 1]
    to_id = meta[season] # we should not include this one
    season_matrix = matrix[from_id:to_id, :]
    return search(text, season_matrix, vocabulary)

def main():
    loc = os.path.join(os.getcwd(),'friends')
    paths = paths_from_dir(loc)
    count_matrix = fit_count_matrix(paths)
    np.save('count_matrix.npy', count_matrix)
    tfidf_matrix = tfidf_transform(count_matrix)
    np.save('tf_idf_matrix.npy', tfidf_matrix)
    vocabulary = np.array(vectorizer.get_feature_names())
    vectorizer.set_params(input = 'content')
    freq_list = get_freq_list(count_matrix)
    reverse_index_count_dict = get_reverse_indexation(count_matrix, vocabulary)
    season_data = [x[0] for x in get_meta(paths)]
    meta = [] # indices where there is a change from one season to another
    for i, el in enumerate(season_data):
        if i > 0:
            if el != season_data[i-1]:
                meta.append(i)
##    a) какое слово является самым частотным
    most_frequent_word = get_n_most_frequent(freq_list, vocabulary, n=1)
    print(most_frequent_word)
##    b) какое самым редким
    least_frequent_words = get_n_least_frequent(freq_list, vocabulary)
    print('only some of the least frequent: ' + str(list(least_frequent_words.items())[500:550]))
##    c) какой набор слов есть во всех документах коллекции
    in_all_texts_bool = np.apply_along_axis(lambda x: 1 if not 0 in x else 0, 0, count_matrix)
    in_all_texts = vocabulary[np.argmax(in_all_texts_bool)]
    print(in_all_texts)
##    d) какой сезон был самым популярным у Чендлера? у Моники?
##    e) кто из главных героев статистически самый популярный?


if __name__ == '__main__':
          main()
