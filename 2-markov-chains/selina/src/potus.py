#!/usr/bin/env python3

#%%
import nltk
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk import ngrams
from nltk.lm import Vocabulary
import re
import numpy as np
import math

#%% [markdown]
"""
# Data Acquisition
"""
def get_trump_data():
    df = pd.read_json('../res/TrumpTweets.json')
    df.drop(df[ df['isRetweet'] == 't' ].index , inplace=True)
    df = df.drop('isRetweet', 1)
    return df

def get_biden_data():
    df = pd.read_csv('../res/BidenTweets.csv')
    return df

def get_obama_data():
    df = pd.read_csv('../res/ObamaTweets.csv')
    return df
#%% [markdown]
"""
# Tokenization
"""
# Since the nltk.lm modules will work on tokenized data, implement a 
# tokenization method that strips unnecessary tokens but retains special
# words such as mentions (@...) and hashtags (#...).
# -> write ourselves, also stop-words?
def tokenize(df):    
    # remove special characters
    df['text_clean'] = df['text'].apply(lambda y: re.sub(r'[^A-Za-z0-9#@ ]+', '', str(y)))

    # tokenize
    tk = TweetTokenizer()  
    df['tokenized_text'] = df.apply(lambda row: tk.tokenize(row['text_clean']), axis=1)   

    # to lower
    df['tokenized_text'] = df['tokenized_text'].map(lambda x: list(map(str.lower, x)))
    return df

#%% [markdown]
"""
# Split Data into training and test 
"""
# 1. Prepare all the tweets, partition into training and test sets; select
#    about 100 tweets each, which we will be testing on later.
#    nb: As with any ML task, training and test must not overlap

def split_data(df):
    train_set = df.sample(frac=((len(df)-100)/len(df)), random_state=0)
    test_set = df.drop(train_set.index)
    return train_set, test_set

#%% [markdown]
"""
# Train Data
"""
# 2. Train n-gram models with n = [1, ..., 5] for both Obama, Trump and Biden.
# 2.1 Also train a joint model, that will serve as background model

def train_data(df, n):
    train_data, padded_sents = padded_everygram_pipeline(n, df['tokenized_text'])
    model = Laplace(n)
    model.fit(train_data, padded_sents)
    # Vocabulary(model.counts, unk_cutoff=1)
    return model

def prepare_test_data(df):
    test_sents = []
    for index,row in df.iterrows():
        test_sent=list(ngrams(row['tokenized_text'], 3))
        test_sents.append(test_sent)
    return test_sents

#%% [markdown]
"""
# Classify Author
"""
# 3. Use the log-ratio method to classify the tweets. Trump should be easy to
#    spot; but what about Biden vs. Obama?
def calc_prob(tweet, model, n):
    prob = 1
    for i in range(0, len(tweet)):
        l = list(tweet[i])
        prob = prob * model.score(l[-1], l[:-1])
    return prob

def compare_authors(model1, model2, test_tweet, n):
    prob1 = calc_prob(test_tweet, model1, n)
    prob2 = calc_prob(test_tweet, model2, n)
    test_var = math.log(prob1/prob2)
    return test_var

#%% [markdown]
"""
# Execute Pipelines
## Trump
"""
tweets_trump_df = get_trump_data()
tweets_trump_df = tweets_trump_df.head(6064)
tweets_trump_df = tokenize(tweets_trump_df)
trump_train, trump_test = split_data(tweets_trump_df)
model_trump = train_data(trump_train, 3)
test_data_trump = prepare_test_data(trump_test)
test_data_trump = [x for x in test_data_trump if x]

#%% [markdown]
"""
## Biden
"""
tweets_biden_df = get_biden_data()
tweets_biden_df = tokenize(tweets_biden_df)
biden_train, biden_test = split_data(tweets_biden_df)
model_biden = train_data(biden_train, 3)
test_data_biden = prepare_test_data(biden_test)
#%% [markdown]
"""
## Obama
"""
tweets_obama_df = get_obama_data()
tweets_obama_df = tokenize(tweets_obama_df)
obama_train, obama_test = split_data(tweets_obama_df)
model_obama = train_data(obama_train, 3)
test_data_obama = prepare_test_data(obama_test)

#%% [markdown]
"""
## Comparison of Trump and Biden
"""
scores = []
for i in range(0, len(test_data_trump)):
    scores.append(compare_authors(model_trump, model_biden, test_data_trump[i], 3 ))
if np.mean(scores) > 0:
    print('Author was Trump')
else: 
    print('Author was Biden')
#%% [markdown]
"""
## Comparison of Obama and Biden
"""
scores = []
for i in range(0, len(test_data_biden)):
    scores.append(compare_authors(model_obama, model_biden, test_data_biden[i], 3 ))
if np.mean(scores) > 0:
    print('Author was Obama')
else: 
    print('Author was Biden')
#%% [markdown]
np.mean(scores)
# 3.1 Analyze: At what context length (n) does the system perform best?
#%% [markdown]
"""
# Compute Metrics
"""
# 4. Compute (and plot) the perplexities for each of the test tweets and 
#    models. Is picking the Model with minimum perplexity a better classifier
#    than in 3.?

