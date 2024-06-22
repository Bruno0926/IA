import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Carregar os arquivos
train_data = pd.read_csv('/mnt/data/ReutersGrain-train.csv')
test_data = pd.read_csv('/mnt/data/ReutersGrain-test.csv')

# Visualizar os dados
print(train_data.head())
print(test_data.head())

nltk.download('punkt')
nltk.download('stopwords')

def preprocess(text):
    # Converte para minúsculas
    text = text.lower()
    # Remove números
    text = re.sub(r'\d+', '', text)
    # Remove pontuações
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenização
    tokens = word_tokenize(text)
    # Remoção de stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    # Reune os tokens em uma string
    processed_text = ' '.join(tokens)
    return processed_text

# Aplicar o pré-processamento
train_data['processed_text'] = train_data['text'].apply(preprocess)
test_data['processed_text'] = test_data['text'].apply(preprocess)

# Visualizar os dados pré-processados
train_data[['Text', 'processed_text']].head(), test_data[['Text', 'processed_text']].head()

print(train_data.head())
print(test_data.head())

# Vetorização dos textos
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data['processed_text'])
X_test = vectorizer.transform(test_data['processed_text'])

y_train = train_data['label']
y_test = test_data['label']

# Treinamento do modelo Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)

# Treinamento do modelo SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

# Avaliação dos modelos
nb_accuracy = accuracy_score(y_test, nb_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)

print("Naive Bayes Accuracy: ", nb_accuracy)
print("SVM Accuracy: ", svm_accuracy)

print("Naive Bayes Classification Report:\n", classification_report(y_test, nb_predictions))
print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))

# Verificar as primeiras linhas dos arquivos CSV para identificar problemas
with open('/mnt/data/ReutersGrain-train.csv', 'r') as file:
    lines = file.readlines()
    for line in lines[:5]:
        print(line)

with open('/mnt/data/ReutersGrain-test.csv', 'r') as file:
    lines = file.readlines()
    for line in lines[:5]:
        print(line)

# Carregar os arquivos com o delimitador correto
train_data = pd.read_csv('/mnt/data/ReutersGrain-train.csv', delimiter=';')
test_data = pd.read_csv('/mnt/data/ReutersGrain-test.csv', delimiter=';')

# Visualizar os dados
print(train_data.head())
print(test_data.head())
