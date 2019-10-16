from time import time

from indexing_elmo import SearchELMo

start = time()
elmo = SearchELMo("elmo_matrix.pickle")
print(f"loading took {time() - start} sec")


def search_two():
    start = time()
    print(elmo.search("я так хочу спать нет сил"))
    print(f"searching took {time() - start} sec")
    start = time()
    print(elmo.search("интересно, заработет ли здесь элмо"))
    print(f"searching took {time() - start} sec")


search_two()
