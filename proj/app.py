"""
This module handles the app.
"""

from flask import Flask, request, render_template
from constants import morph, lemmatize, preprocess

app = Flask(__name__, template_folder="templates")

@app.route('/')
def initial():
    return render_template("index.html")

@app.route('/search', methods=['GET'])
def search():
    text = ""
    engine = ""
    metric = []
    metrics_dict = {"TF-IDF" : [0.9, 0.8, 0.5],
               "bm25" : [0.6, 0.4, 0.1],
               "fasttext" : [0.7, 0.6, 0.3],
               "elmo" : [0.4, 0.3, 0.1]}
    if request.args:
        text = preprocess(request.args["query_text"], lemm=True)
        if "engine" in request.args:
            engine = request.args["engine"]
        else:
            engine = "TF-IDF"
        metric = metrics_dict[engine]
    return render_template("index.html", text=text, engine=engine, arr=3, metric=metric)

if __name__ == "__main__":
    app.run()