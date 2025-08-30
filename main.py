import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pandas as pd

dataset = pd.read_csv(r'data\train.csv', index_col=0)

# Lowercase all the words from each entry in the dataset and saving the modified texts in a new column
dataset['preprocessed_text'] = [text.lower() for text in dataset['text']]

# Removing hashtags
dataset['preprocessed_text'] = [re.sub(pattern='#', repl="", string=text) for text in dataset['preprocessed_text']]

# Removing stopwords
en_stopwords = stopwords.words('english')
dataset['preprocessed_text'] = [" ".join([word for word in text.split() if word not in en_stopwords]) for text in dataset['preprocessed_text']]

# Removing punctuation
dataset['preprocessed_text'] = [re.sub(pattern=r"[^\w\s]", repl=" ", string=text) for text in dataset["preprocessed_text"]]

# Tokenization
dataset['preprocessed_text'] = [word_tokenize(text) for text in dataset['preprocessed_text']]

# Lemmatization
lemmatizer = WordNetLemmatizer()

for text in dataset['preprocessed_text']:
    for word in text:
        word = lemmatizer.lemmatize(word)


preprocessed = sum(dataset['preprocessed_text'], [])
unigrams = (pd.Series(nltk.ngrams(preprocessed, 1)).value_counts())
print(unigrams.head(10))

bigrams = (pd.Series(nltk.ngrams(preprocessed, 2)).value_counts())
print(bigrams.head(10))

ngrams_4 = (pd.Series(nltk.ngrams(preprocessed, 4)).value_counts())
print(ngrams_4.head(10))

# TODO: find a way to remove links 