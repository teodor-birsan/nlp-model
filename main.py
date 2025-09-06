import pandas as pd
import re
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from data_processor.data_processor import preprocess_data, load_preprocessed_data, generate_ngrams, pos, ner
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import spacy

spacy.prefer_gpu(0)
nlp = spacy.load('en_core_web_sm')
pd.plotting.register_matplotlib_converters()

def main():
    train_data = load_preprocessed_data(r'data\preprocessed_train_data.csv')
    test_data = load_preprocessed_data(r'data\preprocessed_train_data.csv')
    X = [','.join(map(str, l)) for l in train_data['preprocessed_text']]
    y = train_data['target']
    word_vec = TfidfVectorizer()
    word_vec_fit = word_vec.fit_transform(X)
    word_tfidf = pd.DataFrame(word_vec_fit.toarray(), columns=word_vec.get_feature_names_out())
    word_tfidf.index = train_data.index
    word_tfidf['sentiment_'] = train_data.sentiment.values
    X_train, X_valid, y_train, y_valid = train_test_split(word_tfidf, y)
    encoder = OrdinalEncoder(categories=[['POS', 'NEU', 'NEG']])
    X_train['sentiment_'] = encoder.fit_transform(X_train[['sentiment_']])
    X_valid['sentiment_'] = encoder.transform(X_valid[['sentiment_']])

    model = LogisticRegression()
    model.fit(X_train, y_train)
    validations = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, validations)
    print(accuracy)

if __name__ == '__main__':
    main()