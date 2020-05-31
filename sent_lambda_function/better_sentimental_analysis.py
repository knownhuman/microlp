import nltk
from nltk import data
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import WordNetLemmatizer
import pickle
import sys
from sklearn import svm
from sklearn.linear_model import LogisticRegression,SGDClassifier
classifier_linear = svm.SVC(kernel='linear')

data.path=['nltk_data']

tokenizer=ToktokTokenizer()
stopword_list=nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

def simple_lemmer(text):
    lemmatizer = WordNetLemmatizer()
    word_list = nltk.word_tokenize(text)
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    return lemmatized_output

def remove_stopwords(text):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

if not 'google.colab' in sys.modules:
    model_file = open('better_sa_classifier.pickle', 'rb')
    model = pickle.load(model_file)
    model_file.close()

    vectorizer_file = open('vectorizer.pickle', 'rb')
    vectorizer = pickle.load(vectorizer_file)
    vectorizer_file.close()

def get_sentiment(review):
    review = remove_special_characters(review)
    review = simple_lemmer(review)
    review = remove_stopwords(review)
    review = [review]
    review_vector = vectorizer.transform(review)
    return model.predict(review_vector)
