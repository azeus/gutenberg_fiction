# This code is based on Michael ODonnel's NLP post to get started with NLTK on 
#Medium https://odonnell31.medium.com/nlp-in-python-a-primer-on-nltk-with-project-gutenberg-fcc02be63d9a.

# request the raw text of the book you are intersted in
import requests
import nltk
import regex as re
import pandas as pd
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

def tokenize_text(text: str):
    
    # lowercase the text
    text = text.lower()
    
    # remove punctuation from text
    text = re.sub(r"[^\w\s]", "", text)
    
    # tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # remove stopwords from txt_tokens and word_tokens
    from nltk.corpus import stopwords
    english_stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in english_stop_words]
    
    # return your tokens
    return tokens

def lemmatize_tokens(tokens):
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # return your lemmatized tokens
    return lemmatized_tokens

# return the most common tokens
def return_top_tokens(tokens,
                      top_N = 10):

    # first, count the frequency of every unique token
    word_token_distribution = nltk.FreqDist(tokens)
    
    # next, filter for only the most common top_N tokens
    # also, put this in a dataframe
    top_tokens = pd.DataFrame(word_token_distribution.most_common(top_N),
                              columns=['Word', 'Frequency'])
    
    # return the top_tokens dataframe
    return top_tokens


# return the most common bi-grams
from nltk.collocations import BigramCollocationFinder

def return_top_bigrams(tokens,
                       top_N = 10):
    
    # collect bigrams
    bcf = BigramCollocationFinder.from_words(tokens)
    
    # put bigrams into a dataframe
    bigram_df = pd.DataFrame(data = bcf.ngram_fd.items(),
                             columns = ['Bigram', 'Frequency'])
    
    # sort the dataframe by frequency
    bigram_df = bigram_df.sort_values(by=['Frequency'],ascending = False).reset_index(drop=True)
    
    # filter for only top bigrams
    bigram_df = bigram_df[0:top_N]
    
    # return the bigram dataframe
    return bigram_df

from nltk.sentiment import SentimentIntensityAnalyzer

def return_sentiment_df(tokens):

    # initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # create some counters for sentiment of each token
    positive_tokens = 0
    negative_tokens = 0
    neutral_tokens = 0
    compound_scores = []
        
    # loop through each token
    for token in tokens:
        
        if sia.polarity_scores(token)["compound"] > 0:
            
            positive_tokens += 1
            compound_scores.append(sia.polarity_scores(token)["compound"])
            
        elif sia.polarity_scores(token)["compound"] < 0:
            
            negative_tokens += 1
            compound_scores.append(sia.polarity_scores(token)["compound"])
              
        elif sia.polarity_scores(token)["compound"] == 0:
            
            neutral_tokens += 1
            compound_scores.append(sia.polarity_scores(token)["compound"])
            
    # put sentiment results into a dataframe
    compound_score_numbers = [num for num in compound_scores if num != 0]
    sentiment_df = pd.DataFrame(data = {"total_tokens" : len(tokens),
                                        "positive_tokens" : positive_tokens,
                                        "negative_tokens" : negative_tokens,
                                        "neutral_tokens" : neutral_tokens,
                                        "compound_sentiment_score" : sum(compound_score_numbers) / len(compound_score_numbers)},
                                index = [0])

    # return sentiment_df
    return sentiment_df

print('Enter the url of the book you are interested in')
message = input('> ')
r = requests.get(message)
book = r.text

# first, remove unwanted new line and tab characters from the text
for char in ["\n", "\r", "\d", "\t"]:
    book = book.replace(char, " ")

tokens = tokenize_text(text = book)

from nltk.stem import WordNetLemmatizer
lemmatized_tokens = lemmatize_tokens(tokens = tokens)

top_tokens = return_top_tokens(tokens = lemmatized_tokens,
                               top_N = 10)
print(top_tokens)

# run the return_top_bigrams function and print the results
bigram_df = return_top_bigrams(tokens = lemmatized_tokens,
                               top_N = 10)
print(bigram_df)

sentiment_df = return_sentiment_df(tokens = lemmatized_tokens)
print(sentiment_df)
