from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np


def yeild_text_from_dir(directory):
    for root, dirs, files in os.walk(directory):
        for name in files:
            if os.path.splitext(name)[1] == '.txt':
                with open(os.path.join(root, name), 'r', encoding='utf-8') as f:
                    yield f.read()


def paths_from_dir(directory):
    paths = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if os.path.splitext(name)[1] == '.txt':
                paths.append(os.path.join(root, name))
    return paths

def get_word_frequency_dictionary(arr):
    freq_dict = {}
    word_freq = np.sum(arr, axis=0)
    vocab = vectorizer.get_feature_names()
    for i, word in enumerate(vocab):
        freq_dict[word] = word_freq[i]
    return freq_dict

directory = os.path.join(os.getcwd(),'friends', 'Friends - season 1')
paths = paths_from_dir(directory)


vectorizer = CountVectorizer(input = 'filename', encoding='utf-8')
X = vectorizer.fit_transform(paths)
arr = X.toarray()
##word_freq = np.sum(arr, axis=0)
##print(vectorizer.get_feature_names()[np.argmax(word_freq)])

freq = get_word_frequency_dictionary(arr)

counter = 0
for key in freq:
    if counter < 100:
        print(key, freq[key])
        counter += 1
