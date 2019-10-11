from flask import Flask, request, render_template
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
               "BM25" : [0.6, 0.4, 0.1],
               "fasttext" : [0.7, 0.6, 0.3],
               "elmo" : [0.4, 0.3, 0.1]}
    if request.args:
        text = request.args["query_text"]
        if "engine" in request.args:
            engine = request.args["engine"]
        else:
            engine = "TF-IDF"
        metric = metrics_dict[engine]
    return render_template("index.html", text=text, engine=engine, arr=3, metric=metric)

if __name__ == "__main__":
    app.run()