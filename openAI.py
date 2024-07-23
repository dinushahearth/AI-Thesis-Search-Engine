import openai
from preprocessData import PreprocessData
from  dataextract import FileRetrieve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

file_retrieve = FileRetrieve()
documents = file_retrieve.get_documents()
preprocess_data = PreprocessData(documents)
processed_documents = preprocess_data.preprocess_documents()


class openAIcls:
    def __init__(self,openAPIkey):
        self.key = openAPIkey

    def generate_summary(self,text):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Summarize the following text:"},
                {"role": "user", "content": text}
            ],
            max_tokens=150
        )
        summary = response['choices'][0]['message']['content'].strip()
        return summary
    
    def summary(self):
        summaries = [self.generate_summary(doc) for doc in processed_documents]
        return summaries

    def calculate_relevance_scores(self,query):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_documents)
        query_vec = vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        return cosine_similarities


