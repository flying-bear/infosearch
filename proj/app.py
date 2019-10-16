"""
This module handles the app.
"""
import tensorflow as tf

from flask import Flask, request, render_template

from indexing_TF_IDF import SearchTfidf
from indexing_bm25 import SearchBM25
from indexing_fasttext import SearchFastText
from indexing_elmo import SearchELMo

app = Flask(__name__, template_folder="templates")
tf_idf = SearchTfidf("lemmatized_normalized_tf_idf_matrix.pickle")
bm = SearchBM25("lemmatized_normalized_bm25_matrix.pickle")
ft = SearchFastText("lemmatized_fasttext_matrix.pickle")


@app.route('/')
def initial():
    return render_template("index.html")


@app.route('/search', methods=['GET'])
def search(n=5):
    text = ""
    engine = ""
    metrics = []
    if request.args:
        text = request.args["query_text"]
        if "engine" in request.args:
            engine = request.args["engine"]
        else:
            engine = "tf-idf"
        if engine == "tf-idf":
            metrics = tf_idf.search(text, n=n)
        elif engine == "bm25":
            metrics = bm.search(text, n=n)
        elif engine == "fasttext":
            metrics = ft.search(text, n=n)
        elif engine == "elmo":
            tf.reset_default_graph()
            el = SearchELMo("elmo_matrix.pickle")
            metrics = el.search(text, n=n)
        if not metrics[0][0]:
            return render_template("index.html", text=text, engine=engine)
    return render_template("index.html", text=text, engine=engine, n=n, metrics=metrics)


if __name__ == "__main__":
    app.run()