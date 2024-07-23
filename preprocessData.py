import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from  dataextract import FileRetrieve

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

class PreprocessData:
    def __init__(self, documents):
        self.documents = documents

    @staticmethod
    def preprocess_text(text):
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens if token.isalpha()]
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(tokens)

    def preprocess_documents(self):
        return [self.preprocess_text(doc) for doc in self.documents]




