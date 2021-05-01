#!/usr/bin/env python3
# The solution to this assignment was created by Selina Feitl and Alexander Birkner
# %%
# imports
import nltk
from nltk.lm.models import Laplace
from nltk.util import flatten, ngrams
import re
import json
import csv
import os
from nltk.lm import Vocabulary
from nltk.stem.snowball import SnowballStemmer
import random
import math
from nltk.lm import MLE
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# %%
# 0. Before you get started, make sure to download the Obama and Trump twitter
#    archives.

# Since the nltk.lm modules will work on tokenized data, implement a 
# tokenization method that strips unnecessary tokens but retains special
# words such as mentions (@...) and hashtags (#...).

def read_texts_from_json(file_path, text_of_tweet_entry):
    res_directory = os.path.dirname(__file__) + '/../res/'
    text_of_tweets = []
    with open(res_directory + file_path, encoding='utf-8') as json_file:
        data = json.load(json_file)
        print('Number of Tweets:', len(data))
        for tweet in data:
            if tweet['isRetweet'] == "f":
                text_of_tweets.append(tweet[text_of_tweet_entry])

    return text_of_tweets

def read_texts_from_csv(file_path, text_of_tweet_entry):
    res_directory = os.path.dirname(__file__) + '/../res/'
    text_of_tweets = []
    with open(res_directory + file_path, encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = 0
        for row in csv_reader:
            text_of_tweets.append(row[text_of_tweet_entry])
            rows = rows + 1

        print('Number of Tweets:', rows)
    return text_of_tweets

def tokenize_texts(input_texts):
    tokenized_tweets = []
    for text in input_texts:
        tokens = tokenize(text)
        tokenized_tweets.append(tokens)

    return tokenized_tweets

# %%
# Preprocessing methods
# Regular Expressions from: https://medium.com/analytics-vidhya/tokenization-building-a-tokenizer-and-a-sentencizer-c19a00393c19
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
    stemmer = SnowballStemmer(language='english')
    input_to_tokenize = ' '.join([stemmer.stem(word) for word in input_to_tokenize.split(' ')])

    # Tokenization
    input_to_tokenize = re.sub(split_characters, delimiter_token, input_to_tokenize)
    list_of_token_strings = [x.strip() for x in input_to_tokenize.split(delimiter_token) if x.strip() !="" and len(x.strip()) > 1]
   
    return list_of_token_strings

def get_train_and_test_tokens_from_file(file_name, text_of_tweet_entry, number_train_data=5500, number_test_data=100):
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
    tokens['train'] = tokenized_tweets[number_test_data:number_test_data+number_train_data]

    return tokens

# %%
# 1. Prepare all the tweets, partition into training and test sets; select
#    about 100 tweets each, which we will be testing on later.
#    nb: As with any ML task, training and test must not overlap
trump_data = get_train_and_test_tokens_from_file('TrumpTweets.json', 'text')
obama_data = get_train_and_test_tokens_from_file('ObamaTweets.csv', 'text')
biden_data = get_train_and_test_tokens_from_file('BidenTweets.csv', 'text')

print('Sample Output')
print(trump_data['train'][1])
print(obama_data['train'][1])
print(biden_data['train'][1])

# %%
# 2. Train n-gram models with n = [1, ..., 5] for both Obama, Trump and Biden.
# 2.1 Also train a joint model, that will serve as background model

# alternative ngram management
# def get_dict_of_ngrams(training_data, n_grams=0):
#     if n_grams == 0:
#         n_grams = {1:[], 2:[], 3:[], 4:[], 5:[]}

#     for sentencized_tweet in training_data:
#         for tokenized_sentence in sentencized_tweet:
#             n_grams[1].append(list(ngrams(tokenized_sentence, 1)))
#             n_grams[2].append(list(ngrams(tokenized_sentence, 2)))
#             n_grams[3].append(list(ngrams(tokenized_sentence, 3)))
#             n_grams[4].append(list(ngrams(tokenized_sentence, 4)))
#             n_grams[5].append(list(ngrams(tokenized_sentence, 5)))
    
#     return n_grams

def get_ngrams(training_data, n_grams=0):
    if n_grams == 0:
        n_grams = []

    for sentencized_tweet in training_data:
        for tokenized_sentence in sentencized_tweet:
            l = []
            l.append(list(ngrams(tokenized_sentence, 1)))
            l.append(list(ngrams(tokenized_sentence, 2)))
            l.append(list(ngrams(tokenized_sentence, 3)))
            l.append(list(ngrams(tokenized_sentence, 4)))
            l.append(list(ngrams(tokenized_sentence, 5)))
            flat_list = [item for sublist in l for item in sublist]
            n_grams.append(flat_list)
    
    return n_grams

def gen_vocab(training_data):
    vocab = list(flatten(training_data))            
    return Vocabulary(vocab, unk_cutoff=1)

# %%
trump_ngrams = get_ngrams(trump_data['train'])
trump_vocab = gen_vocab(trump_data['train'])

obama_ngrams = get_ngrams(obama_data['train'])
obama_vocab = gen_vocab(obama_data['train'])

biden_ngrams = get_ngrams(biden_data['train'])
biden_vocab = gen_vocab(biden_data['train'])

joint_ngrams = get_ngrams(trump_data['train'])

joint_ngrams = get_ngrams(obama_data['train'], joint_ngrams)
joint_ngrams = get_ngrams(biden_data['train'], joint_ngrams)
joint_vocab = list(flatten(trump_data['train'], obama_data['train'], biden_data['train']))
joint_vocab = Vocabulary(joint_vocab, unk_cutoff=1)

print('Sample Output')
print(trump_ngrams[0])
print(joint_ngrams[0])

# %%
def train_model(n, training_data, vocab):
    lm = Laplace(n)
    lm.fit(training_data, vocab)
    return lm

def get_models_for_president(n_grams, vocab):
    models = {}
    for i in range(1,6):
        models[i] = train_model(i, n_grams, vocab)
        print('Model for ', i , '-grams trained')
    return models

# %%
trump_models = get_models_for_president(trump_ngrams, trump_vocab)
obama_models = get_models_for_president(obama_ngrams, obama_vocab)
biden_models = get_models_for_president(biden_ngrams, biden_vocab)
joint_models = get_models_for_president(joint_ngrams, joint_vocab)

# %%
# 3. Use the log-ratio method to classify the tweets. Trump should be easy to
#    spot; but what about Biden vs. Obama?
# 3.1 Analyze: At what context length (n) does the system perform best?
def calc_score_of_tweet(tweet, lm_model, n):
    scores = []
    for sentence in tweet:
        for i in range(1, len(sentence)-1):
            score = 0
            for j in range(n):
                if i-j > 0:
                    score = score + lm_model.logscore(sentence[i], sentence[i-j:i])
            scores.append(score)
    return scores

def compare_authors_for_tweet(model1, model2, test_tweet, n):
    scores1 = calc_score_of_tweet(test_tweet, model1, n)
    scores2 = calc_score_of_tweet(test_tweet, model2, n)

    if np.mean(scores1) > np.mean(scores2):
        return 1
    else:
        return 2

def compare_authors_for_test_set(models1, models2, test_set, n):
    correct = 0
    for tweet in test_set:
        result = compare_authors_for_tweet(models1[n], models2[n], tweet, n)
        if result == 1:
            correct += 1
    return correct

# %%
def compare(name1: str, name2: str, models1: dict, models2: dict, test_data1, test_data2):
    print(name1,'vs.', name2)
    print('Test data of', name1)
    for i in range(1,6):
        print('Context lenght:', i)
        print(compare_authors_for_test_set(models1, models2, test_data1, i))
    print('Test data of ' + name2)
    for i in range(1,6):
        print('Context lenght:', i)
        print(compare_authors_for_test_set(models2, models1, test_data2, i))

# %%
# Trump vs. Biden:
compare('Trump', 'Biden', trump_models, biden_models, trump_data['test'], biden_data['test'])

# %%
# Trump vs. Obama:
compare('Trump', 'Obama', trump_models, obama_models, trump_data['test'], obama_data['test'])

# %%
# Biden vs. Obama:
compare('Biden', 'Obama', biden_models, obama_models, biden_data['test'], obama_data['test'])

# %%
def compare_joint(name: str, models: dict, joint_models: dict, test_data):
    print(name,'vs. Joint Model')
    print('Test data of', name)
    for i in range(1,6):
        print('Context lenght:', i)
        print(compare_authors_for_test_set(models, joint_models, test_data, i))
# %%
# Biden vs. Joint
compare_joint('Biden', biden_models, joint_models, biden_data['test'])

# %%
# Obama vs. Joint
compare_joint('Obama', obama_models, joint_models, obama_data['test'])

# %%
# Trump vs. Joint
compare_joint('Trump', trump_models, joint_models, trump_data['test'])

# %%
# 4. Compute (and plot) the perplexities for each of the test tweets and 
#    models. Is picking the Model with minimum perplexity a better classifier
#    than in 3.?

def get_perplexities_of_tweet_set_multi_models(tweet_ngrams, n, models1, models2, models3):
    perplexities = []
    for tweets in tweet_ngrams:
        for i in range(0, len(tweets)):
            tweet=list(ngrams(tweets[i], n))
            if len(tweet) > 0:
                perplexities_of_tweet = []
                perplexities_of_tweet.append(models1[n].perplexity(tweet))
                perplexities_of_tweet.append(models2[n].perplexity(tweet))
                perplexities_of_tweet.append(models3[n].perplexity(tweet))
            perplexities.append(perplexities_of_tweet)
    return perplexities

def get_correct_classifications(perplexities, index_expected):
    correct_amount = 0
    print('Total Amount: ' + str(len(perplexities)))
    for tweet_perplexity in perplexities:
        min_perplexity_index = tweet_perplexity.index(min(tweet_perplexity))
        if index_expected == min_perplexity_index:
            correct_amount = correct_amount + 1
    return correct_amount

def get_perplexities_of_tweet_set(tweet_ngrams, n, models):
    perplexities = []
    for tweet in tweet_ngrams:
        sentence_perplex = []
        for i in range(0, len(tweet)):
            sent=list(ngrams(tweet[i], n))
            if len(sent) > 0:
                sentence_perplex.append(models[n].perplexity(sent))
        perplexities.append(np.mean(sentence_perplex))
    return perplexities

# %%
for i in range(1,6):
    print('Correct classifications for context length: ', i)
    perplexities_trump_test_data = get_perplexities_of_tweet_set_multi_models(trump_data['test'], i, trump_models, biden_models, obama_models)
    print('Trump', get_correct_classifications(perplexities_trump_test_data, 0))

    perplexities_biden_test_data = get_perplexities_of_tweet_set_multi_models(biden_data['test'], i, trump_models, biden_models, obama_models)
    print('Biden', get_correct_classifications(perplexities_biden_test_data, 1))

    perplexities_obama_test_data = get_perplexities_of_tweet_set_multi_models(obama_data['test'], i, trump_models, biden_models, obama_models)
    print('Obama', get_correct_classifications(perplexities_obama_test_data, 2))

# %%
trumps_perplexities = get_perplexities_of_tweet_set(obama_data['test'], 2, trump_models)
bidens_perplexities = get_perplexities_of_tweet_set(obama_data['test'], 2, biden_models)
obamas_perplexities = get_perplexities_of_tweet_set(obama_data['test'], 2, obama_models)

plt.plot(trumps_perplexities, label='Trump') 
plt.plot(bidens_perplexities, label='Biden') 
plt.plot(obamas_perplexities, label='Obama') 
plt.legend()
plt.show()
# %%