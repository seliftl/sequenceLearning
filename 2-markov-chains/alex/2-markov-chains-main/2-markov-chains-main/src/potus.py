#!/usr/bin/env python3

# we will be using nltk.lm

# imports
# %%
import nltk
from nltk.util import flatten, ngrams
from nltk.lm.models import Laplace
import re
import json
import csv
import os
from nltk.stem.snowball import SnowballStemmer
import random
from nltk.lm import MLE


# 0. Before you get started, make sure to download the Obama and Trump twitter
#    archives.

# Since the nltk.lm modules will work on tokenized data, implement a 
# tokenization method that strips unnecessary tokens but retains special
# words such as mentions (@...) and hashtags (#...).

# %%
# Read methods
def read_texts_from_json(file_path, text_of_tweet_entry):
    res_directory = os.path.dirname(__file__) + '/../res/'
    text_of_tweets = []
    with open(res_directory + file_path, encoding='utf-8') as json_file:
        data = json.load(json_file)
        for tweet in data:
            if tweet['isRetweet'] == "f":
                text_of_tweets.append(tweet[text_of_tweet_entry])

    return text_of_tweets

def read_texts_from_csv(file_path, text_of_tweet_entry):
    res_directory = os.path.dirname(__file__) + '/../res/'
    text_of_tweets = []
    with open(res_directory + file_path, encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            text_of_tweets.append(row[text_of_tweet_entry])

    return text_of_tweets

def tokenize_texts(input_texts):
    tokenized_tweets = []
    for text in input_texts:
        tokens = tokenize(text)
        tokenized_tweets.append(tokens)

    return tokenized_tweets

# %%
# Preprocessing methods
DEFAULT_SENTENCE_BOUNDARIES = ['(?<=[0-9]|[^0-9.])(\.)(?=[^0-9.]|[^0-9.]|[\s]|$)','\.{2,}','\!+','\:+','\?+']

DEFAULT_PUNCTUATIONS = ['(?<=[0-9]|[^0-9.])(\.)(?=[^0-9.]|[^0-9.]|[\s]|$)','\.{2,}',
                        '\!+','\:+','\?+','\,+','\"+', '\”+', '\“+','\//+', r'\(|\)|\[|\]|\{|\}']

def sentencize(raw_input_document, sentence_boundaries = DEFAULT_SENTENCE_BOUNDARIES, delimiter_token='<SPLIT>'):
    working_document = raw_input_document
    punctuation_patterns = sentence_boundaries
    for punct in punctuation_patterns:
        working_document = re.sub(punct, '\g<0>'+delimiter_token, working_document, flags=re.UNICODE)

    # Remove links
    working_document = re.sub(r"http\S+", "", working_document)

    list_of_string_sentences = ['<s> ' + x.strip() + ' </s>' for x in working_document.split(delimiter_token) if x.strip() != ""]
    return list_of_string_sentences

def tokenize(input, punctuation_patterns= DEFAULT_PUNCTUATIONS, split_characters = r'\s|\t|\n|\r', delimiter_token='<SPLIT>'):
    input_to_tokenize = input
    # Whitespace removal
    input_to_tokenize = ' '.join(input_to_tokenize.split())

    # To lower case
    input_to_tokenize = input_to_tokenize.lower()

    # Remove punctuation and numbers
    for punct in punctuation_patterns:
        input_to_tokenize = re.sub(punct, "", input_to_tokenize)

    # Stemming
    #stemmer = SnowballStemmer(language='english')
    #input_to_tokenize = ' '.join([stemmer.stem(word) for word in s.split(' ')])

    # Tokenization
    input_to_tokenize = re.sub(split_characters, delimiter_token, input_to_tokenize)
    list_of_token_strings = [x.strip() for x in input_to_tokenize.split(delimiter_token) if x.strip() !="" and len(x.strip()) > 1]
   
    return list_of_token_strings

def get_train_and_test_tokens_from_file(file_name, text_of_tweet_entry, number_test_data=100):
    tokens = {}
    name, file_extension = os.path.splitext(file_name)

    # Read
    tweet_texts = 0
    if file_extension == '.json':
        tweet_texts = read_texts_from_json(file_name, text_of_tweet_entry)
    elif file_extension == '.csv':
        tweet_texts = read_texts_from_csv(file_name, text_of_tweet_entry)
    else:
        raise Exception('File type not supported')

    # Tokenize
    tokenized_tweets = []
    for tweet_text in tweet_texts:
        sentencized_tweet = sentencize(tweet_text)

        tokenized_sentences = []
        for sentence in sentencized_tweet:
            tokenized_sentence = tokenize(sentence)
            if len(tokenized_sentence) > 2:
                tokenized_sentences.append(tokenized_sentence)

        tokenized_tweets.append(tokenized_sentences)
    
    # Seperate
    random.shuffle(tokenized_tweets)
    tokens['test'] = tokenized_tweets[:number_test_data]
    tokens['train'] = tokenized_tweets[number_test_data:]

    return tokens

# 1. Prepare all the tweets, partition into training and test sets; select
#    about 100 tweets each, which we will be testing on later.
#    nb: As with any ML task, training and test must not overlap

# %%
trump_data = get_train_and_test_tokens_from_file('tweets_01-08-2021.json', 'text')
obama_data = get_train_and_test_tokens_from_file('Tweets-BarackObama.csv', 'Tweet-text')
biden_data = get_train_and_test_tokens_from_file('JoeBidenTweets.csv', 'tweet')

print('Sample Output')
print(trump_data['train'][0])

# 2. Train n-gram models with n = [1, ..., 5] for both Obama, Trump and Biden.
# 2.1 Also train a joint model, that will serve as background model

# %%
def get_dict_of_ngrams(training_data, n_grams=0):
    if n_grams == 0:
        n_grams = {'unigrams':[],'bigrams':[],'trigrams':[],'fourgrams':[],'fivegrams':[]}

    for sentencized_tweet in training_data:
        for tokenized_sentence in sentencized_tweet:
            n_grams['unigrams'].extend(list(ngrams(tokenized_sentence, 1)))
            n_grams['bigrams'].extend(list(ngrams(tokenized_sentence, 2)))
            n_grams['trigrams'].extend(list(ngrams(tokenized_sentence, 3)))
            n_grams['fourgrams'].extend(list(ngrams(tokenized_sentence, 4)))
            n_grams['fivegrams'].extend(list(ngrams(tokenized_sentence, 5)))
    
    return n_grams

def gen_vocab(training_data):
    vocab = nltk.flatten(training_data)
    return vocab

# %%
trump_ngrams = get_dict_of_ngrams(trump_data['train'])
trump_vocab = gen_vocab(trump_data['train'])

obama_ngrams = get_dict_of_ngrams(obama_data['train'])
obama_vocab = gen_vocab(obama_data['train'])

biden_ngrams = get_dict_of_ngrams(biden_data['train'])
biden_vocab = gen_vocab(biden_data['train'])

joint_ngrams = get_dict_of_ngrams(trump_data['train'])
joint_ngrams = get_dict_of_ngrams(obama_data['train'], joint_ngrams)
joint_ngrams = get_dict_of_ngrams(biden_data['train'], joint_ngrams)

print('Sample Output')
print(trump_ngrams['unigrams'][0])
print(trump_ngrams['bigrams'][0])
print(trump_ngrams['trigrams'][0])
print(trump_ngrams['fourgrams'][0])
print(trump_ngrams['fivegrams'][0])

fdist = nltk.FreqDist(trump_ngrams['bigrams'])
print("Most common bigrams: ")
print(fdist.most_common(10))



# %%
lm_trump = Laplace(2)
len(lm_trump.vocab)
lm_trump.fit(trump_ngrams['bigrams'], trump_vocab)
len(lm_trump.vocab)


# %%

# 3. Use the log-ratio method to classify the tweets. Trump should be easy to
#    spot; but what about Biden vs. Trump?
# 3.1 Analyze: At what context length (n) does the system perform best?


# 4. Compute (and plot) the perplexities for each of the test tweets and 
#    models. Is picking the Model with minimum perplexity a better classifier
#    than in 3.?

# %%


# %%
