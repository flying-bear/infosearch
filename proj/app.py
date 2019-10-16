"""
This module handles the app.
"""

import tensorflow as tf

from flask import Flask, request, render_template

from constants import logger
from constants import path_tfidf_matrix, path_bm25_matrix, path_fasttext_matrix, path_elmo_matrix

from indexing_TF_IDF import SearchTfidf
from indexing_bm25 import SearchBM25
from indexing_fasttext import SearchFastText
from indexing_elmo import SearchELMo

logger.info("________________new app run__________________")
app = Flask(__name__, template_folder="templates")

tf_idf = SearchTfidf(path_tfidf_matrix)
bm = SearchBM25(path_bm25_matrix)
ft = SearchFastText(path_fasttext_matrix)


@app.route('/')
def initial():
    return render_template("index.html")


@app.route('/search', methods=['GET'])
def search():
    try:  # log exceptions
        if request.args:
            if "n" in request.args:  # if n specified
                n = int(request.args["n"])
            else:  # n=10 by default
                n = 10
            metrics = []
            text = request.args["query_text"]  # REQUIRED
            if "engine" in request.args:  # if engine specified
                engine = request.args["engine"]
            else:  # tf-idf by default
                engine = "tf-idf"
            if engine == "tf-idf":
                metrics = tf_idf.search(text, n=n)
            elif engine == "bm25":
                metrics = bm.search(text, n=n)
            elif engine == "fasttext":
                metrics = ft.search(text, n=n)
            elif engine == "elmo":
                tf.reset_default_graph()
                el = SearchELMo(path_elmo_matrix)
                metrics = el.search(text, n=n)
            if not metrics[0][0]:
                logger.info(f"nothing was found for '{text}' in {engine} model")
                return render_template("index.html", text=text, engine=engine)
            metrics = [item for item in metrics if item[0]]
            if n != len(metrics):
                logger.info(f"expected to find {n} results, but found only {len(metrics)}")
                n = len(metrics)
            return render_template("index.html", text=text, engine=engine, n=n, metrics=metrics)
        else:
            return render_template("index.html")
    except Exception as ex:  # catch and log and do not embarrass yourself
        logger.critical(f"app crashed with exception '{ex}'")
        return render_template("index.html", exception=ex)

if __name__ == "__main__":
    app.run()