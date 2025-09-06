import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
import spacy
from transformers import pipeline
import numpy as np
from nltk.corpus import stopwords

def insert_columns(dataset: pd.DataFrame, columns: list[tuple[int, str]]) -> pd.DataFrame:
    for column in columns:
        dataset.insert(loc=column[0], column=column[1], value=np.nan)
    print(f"Inserted {len(columns)} columns.\n")
    return dataset


def copy_column_data(dataset: pd.DataFrame, source: str, destination: str) -> pd.DataFrame:
    dataset[destination] = dataset[source]
    print(f"Copied data from {source} to {destination}.\n")
    return dataset


def lower_case_text_columns(dataset: pd.DataFrame, columns: list[str],) -> pd.DataFrame:
    for column in columns:
        try:
            dataset[column] = [text.lower() for text in dataset[column]]
        except Exception as e:
            print(e)
    print(f"Lowercased words in {columns}.\n")
    return dataset

def remove_pattern(dataset: pd.DataFrame, pattern: str, column: str,) -> pd.DataFrame:
    try:
        dataset[column] = [re.sub(pattern=pattern, repl="", string=text) for text in dataset[column]]
    except Exception as e:
        print(e)
    print(f"Removed pattern: {pattern}\n")
    return dataset

def remove_words(dataset: pd.DataFrame, words_list: list[str], column: str,) -> pd.DataFrame:
    dataset[column] = [" ".join([word for word in text.split() if word not in words_list]) for text in dataset[column]]
    print("Removed words.\n")
    return dataset

def tokenize_text(dataset: pd.DataFrame, column: str,) -> pd.DataFrame:
    dataset[column] = [word_tokenize(text) for text in dataset[column]]
    print(f"Tokenized text from column: {column}\n")
    return dataset

def lemmatize_words(dataset: pd.DataFrame, column: str,) -> pd.DataFrame:
    lemmitizer = WordNetLemmatizer()
    final_texts = []
    for text in dataset[column]:
        lemmatized_text = []
        for word in text:
            lemmatized_text.append(lemmitizer.lemmatize(word))
        final_texts.append(lemmatized_text)
    dataset[column] = final_texts
    print(f"Lemmatiezed text from column: {column}\n")
    return dataset

def stem_words(dataset: pd.DataFrame, column: str,) -> pd.DataFrame:
    stemmer = PorterStemmer()
    dataset[column] = [stemmer.stem(word) for text in dataset[column] for word in text]
    print(f"Stemmed text from column: {column}\n")
    return dataset


def generate_ngrams(dataset: pd.DataFrame, column: str, n: int, save: bool = False) -> pd.Series | None:
    data = sum(dataset[column], [])
    ngram = (pd.Series(nltk.ngrams(data, n)).value_counts())
    if save:
        ngram.to_csv(rf"data\ngram_{n}.csv")
        print(rf"Generated and saved ngrams with {n} words to data\ngram_{n}.csv", '\n')
    else:
        print(ngram.head(10))
        return ngram
    return 
            
def sentiment_analysis(dataset: pd.DataFrame, text_column: str, label_column: str) -> pd.DataFrame:
    sentiment_pipeline = pipeline('sentiment-analysis', model='finiteautomata/bertweet-base-sentiment-analysis', device=0)
    transformer_labels = []
    for text in dataset[text_column].values:
        sentiment_list = sentiment_pipeline(text, batch_size=32)
        sentiment_label = [sent['label'] for sent in sentiment_list]
        transformer_labels.append(sentiment_label)
    dataset[label_column] = sum(transformer_labels, [])
    print("Added sentiment labels to each entry.\n")
    return dataset

def ner(dataset: pd.DataFrame, column: str, nlp: spacy.language.Language) -> pd.DataFrame:
    ner_df = pd.DataFrame(columns=['word', 'ner_tag'])
    for text in dataset[column]:
        spacy_doc = spacy.tokens.doc.Doc(nlp.vocab, words=text)
        doc = nlp(spacy_doc)
        for word in doc.ents:
            ner_df = pd.concat([ner_df, pd.DataFrame.from_records([
                {'word': word.text, 'ner_tag': word.label_}
            ])])
    return ner_df

def pos(dataset: pd.DataFrame, column: str, nlp: spacy.language.Language) -> pd.DataFrame:
    pos_df = pd.DataFrame(columns=['token', 'pos_tag'])
    for text in dataset[column]:
        doc = spacy.tokens.doc.Doc(nlp.vocab, words = text)
        spacy_doc = nlp(doc)
        for token in spacy_doc:
            pos_df = pd.concat([pos_df, pd.DataFrame.from_records([
                {'token': token.text, 'pos_tag': token.pos_}
            ])])
    return pos_df



def preprocess_data(dataset: pd.DataFrame, save_to_file: bool = False, filename: str = ""):
    words = stopwords.words('english')
    new_dataset =(dataset.pipe(insert_columns, columns=[(3, 'preprocessed_text'), (4, 'sentiment')]) # creating new columns
            .pipe(copy_column_data, source='text', destination='preprocessed_text') 
            .pipe(lower_case_text_columns, columns=['preprocessed_text']) # lower case all the words
            .pipe(remove_pattern, pattern=r"https?://\S", column='preprocessed_text') # remove all links
            .pipe(remove_pattern, pattern=r"[^\s\w]", column='preprocessed_text') # remove punctuation
            .pipe(sentiment_analysis, text_column='preprocessed_text', label_column='sentiment') # sentiment analysis
            .pipe(remove_words, words_list=words, column='preprocessed_text') # remove stop words
            .pipe(tokenize_text, column='preprocessed_text') # tokenize text
            .pipe(lemmatize_words, column='preprocessed_text') # lemmiteze words
     )
    if save_to_file:
        if filename == "":
            filename = "preprocessed_dataset"
        new_dataset.to_csv(fr"data\{filename}.csv")
    return new_dataset


def load_preprocessed_data(path: str) -> pd.DataFrame:
    dataset = pd.read_csv(path, index_col=0)
    dataset['preprocessed_text'] = [re.sub(f"[\[\]']", repl="", string=text) for text in dataset['preprocessed_text']]
    dataset['preprocessed_text'] = [text.split(', ') for text in dataset['preprocessed_text']]
    return dataset
