from flask import Flask, render_template, request, jsonify
from openAI import openAIcls
import openai
from  dataextract import FileRetrieve

openai.api_key = '<open API key>'
openAI = openAIcls(openai.api_key)
summaries = openAI.summary()
file_retrieve = FileRetrieve()
documents = file_retrieve.get_documents()

app = Flask(__name__)

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    relevance_scores = openAI.calculate_relevance_scores(query)
    results = [{"summary": summaries[i], "relevance_score": relevance_scores[i]} for i in range(len(documents))]
    results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
    
    return render_template('results.html', results=results)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
