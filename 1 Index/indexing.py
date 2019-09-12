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

## TODO:
##    - read from zip
##    - reusable
##    - pep8

import os
import pandas as pd
import numpy as np
import re
raw_json_path = 'friends_raw.json'


def text_strip_split(text):
    low = text.lower()
    stripped = re.sub('\.|,|#|$|%|\\|\'|\(|\)|-|\+|\*|/|\:|;|<|>|=|\?|\[|\]|@|^|_|`|{|}|~', '', low)
    words = stripped.split()
    return words


def read_from_dir_to_pd(directory, column_names): # potentially reusable
    raw_data = pd.DataFrame()
    raw_data = pd.DataFrame(columns=column_names)
    for root, dirs, files in os.walk(directory):
        for name in files:
            ## мерзкая конкретика - preprocessing: splitting name
            ep_num, ep_name = name.split(' - ')[1:]
            ep_name = ep_name.strip('ru.txt')
            season, episode = ep_num.split('x')
            ## конец мерзкой конкретики
            with open(os.path.join(root, name), 'r', encoding='utf-8') as f:
                text = f.read()
            raw_data = raw_data.append(dict(zip(column_names, [ep_name, season, episode, text, text_strip_split(text)])), ignore_index=True) # немного мерзкой конкретики - values
    return raw_data


def get_and_save_df():
    column_names = ['name', 'season', 'episode', 'text', 'words']
    raw = read_from_dir_to_pd(os.path.join(os.getcwd()+'/friends'), column_names)
    raw.to_json(raw_json_path, force_ascii=False)


def main():
    get_and_save_df()
    raw = pd.read_json(raw_json_path)
    print(raw.head())


if __name__ == '__main__':
    main()
