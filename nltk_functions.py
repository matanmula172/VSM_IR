import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

my_stopwords = set(stopwords.words('english'))


# This function tokenizes a given text using nltk, while removing punctuations
def tokenize_text_no_punctuation(text):
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    return tokenizer.tokenize(text)


# This function removes stop words from a tokenized text
def remove_stop_words(tokenized_text):
    new_tokenized_text = []
    for token in tokenized_text:
        if token not in my_stopwords:
            new_tokenized_text.append(token)
    return new_tokenized_text


# This function stems a tokenized text
def stem_tokenized_text(tokenized_text):
    stemmer = PorterStemmer()
    stemmed_tokens = []
    for token in tokenized_text:
        stemmed_tokens.append(stemmer.stem(token))

    return stemmed_tokens
