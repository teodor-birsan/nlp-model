import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy
from spacy import tokenizer
import re
import pandas as pd

dataset = pd.read_csv(r'data\train.csv', index_col=0)
nlp = spacy.load('en_core_web_sm')

# Lowercase all the words from each entry in the dataset and saving the modified texts in a new column
dataset['preprocessed_text'] = [text.lower() for text in dataset['text']]

# Removing hashtags
dataset['preprocessed_text'] = [re.sub(pattern='#', repl="", string=text) for text in dataset['preprocessed_text']]

# Remove links 
dataset['preprocessed_text'] = [re.sub(pattern=r"https?://\S+", repl="", string=text) for text in dataset['preprocessed_text']]

# Removing punctuation
dataset['preprocessed_text'] = [re.sub(pattern=r"[^\w\s]", repl=" ", string=text) for text in dataset["preprocessed_text"]]

# Removing stopwords
en_stopwords = stopwords.words('english')
dataset['preprocessed_text'] = [" ".join([word for word in text.split() if word not in en_stopwords]) for text in dataset['preprocessed_text']]

# Named entity recognition
ner_df = pd.DataFrame(columns=['word', 'ner_tag'])
for text in dataset['preprocessed_text']:
    spacy_doc = nlp(text)
    for word in spacy_doc.ents:
        ner_df = pd.concat([ner_df, pd.DataFrame.from_records([
            {'word': word.text, 'ner_tag': word.label_}
        ])])

print(ner_df.head(10))

# Tokenization
dataset['preprocessed_text'] = [word_tokenize(text) for text in dataset['preprocessed_text']]

# Lemmatization
lemmatizer = WordNetLemmatizer()

for text in dataset['preprocessed_text']:
    for word in text:
        word = lemmatizer.lemmatize(word)

# Parts of speech tagging
pos_df = pd.DataFrame(columns=['token', 'pos_tag'])
for text in dataset['preprocessed_text']:
    spacy_doc = spacy.tokens.doc.Doc(
        nlp.vocab, words = text
    )
    spacy_doc = nlp(spacy_doc)
    for token in spacy_doc:
        pos_df = pd.concat([pos_df, pd.DataFrame.from_records([
            {'token': token.text, 'pos_tag': token.pos_}
        ])])

print(pos_df.head(10))

preprocessed = sum(dataset['preprocessed_text'], [])

bigrams = (pd.Series(nltk.ngrams(preprocessed, 2)).value_counts())
print(bigrams.head(10))

ngrams_4 = (pd.Series(nltk.ngrams(preprocessed, 4)).value_counts())
print(ngrams_4.head(10))
