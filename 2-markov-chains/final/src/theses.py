#!/usr/bin/env python3
#%%
# Imports
import numpy as np
import nltk
import io
import re
from nltk import ngrams
from nltk.lm import Vocabulary
from nltk.lm.models import Laplace

# %%
# 0. Before you get started, make sure to download the `theses.txt` data set.
def load_data() -> list:
    file = io.open("../res/theses.txt", mode="r", encoding="utf-8")
    theses = file.readlines()
    theses = [x.strip() for x in theses] 
    return theses
theses = load_data()

# %%
# 1. Spend some time on pre-processing. How would you handle hyphenated words
#    and abbreviations/acronyms?

def prepare_data(theses: list) -> list:
    preprocessed_theses = []
    for title in theses:        
        # Remove punctuations and numbers except dashes in-between words
        # title = re.sub('[^a-zäöüA-ZÄÖÜ-]', ' ', title)
        title = re.sub('[- ]', ' ', title)
        title = re.sub('[?!.:,()/]``''', ' ', title)
        # Remove multiple spaces
        title = re.sub(r'\s+', ' ', title)
        preprocessed_theses.append(title.lower())
    return preprocessed_theses
preprocessed_data = prepare_data(theses)

# %%
# Tokenization
def tokenize_text(text):  
    tokens = nltk.word_tokenize(text)
    tokens.insert(0, '<s>')
    tokens.append('</s>')
    return tokens


titles_as_tokens = [tokenize_text(title) for title in preprocessed_data]
print('Sample output tokenization')
print(titles_as_tokens[0])

# %%
# 2. Train n-gram models with n = [1, ..., 5]. What about <s> and </s>?
# N-Gram Generation
def get_ngrams(training_data, n_grams=0):
    n_grams = []
    for title in training_data:
        l = []
        for i in range(1,6):
            l.append(list(ngrams(title, i)))
        flat_list = [item for sublist in l for item in sublist]
        # print(flat_list)
        n_grams.append(flat_list)
    return n_grams

n_grams = get_ngrams(titles_as_tokens)
print('Sample output ngrams')
print(n_grams[0])

# %%
# Vocab Generation
vocab = Vocabulary(list(nltk.flatten(titles_as_tokens)))
# %%
# Training of Models
def train_model(n, training_data, vocab):
    lm = Laplace(n)
    lm.fit(training_data, vocab)
    return lm

def get_models(n_grams, vocab):
    models = {}
    for i in range(1,6):
        models[i] = train_model(i, n_grams, vocab)
        print('Model for ', i, '-grams trained')
    return models

language_models = get_models(n_grams, vocab) 

# %%
# 3. Write a generator that provides thesis titles of desired length. Please
#    do not use the available `lm.generate` method but write your own.
#    nb: If you fix the seed in numpy.random.choice, you get reproducible 
#        results.
# 3.1 How can you incorporate seed words?
# 3.2 How do you handle </s> tokens (w.r.t. the desired length?)

def generate(n: int, length: int, language_models:dict, seed: str='<s>') -> str:
    title = seed.split()
    if seed == '<s>':
        length += 1
    
    while len(title) < length:
        next_word = ""

        next_word = get_word_from_n_gram_model(n, title, language_models)
        if next_word != '</s>' and next_word != "":
            title.append(next_word)

    return ' '.join(title)

def calculate_model_order_to_use(n, cur_title):
    if len(cur_title) >= n-1:
        return n
    else:
        return n - len(cur_title)

def get_word_from_one_gram_model(language_models):
    res = language_models[1].counts[1].items()
    possible_words = []
    weights = []
    summed_weights = 0

    for word_with_count in res:
        possible_words.append(word_with_count[0])
        weights.append(word_with_count[1])
        summed_weights += word_with_count[1]

    scaled_weights = []
    for weight in weights:
        scaled_weights.append(weight/summed_weights)

    next_word = np.random.choice(possible_words, 1, scaled_weights)[0]
    return next_word

def get_word_from_n_gram_model(n, cur_title, language_models):
    cur_n = calculate_model_order_to_use(n, cur_title)

    if cur_n == 1:
        return get_word_from_one_gram_model(language_models)
    
    last_n_minus_one_words = tuple(cur_title[-(cur_n-1):])
    following_words_with_count = language_models[cur_n].counts[last_n_minus_one_words].items()

    #remove </s> -tags to avoid endless looping
    following_words_with_count = [i for i in following_words_with_count if i[0] != '</s>']

    possible_words = []    
    weights = []
    summed_weights = 0
    for word_with_count in following_words_with_count:
        possible_words.append(word_with_count[0])
        weights.append(word_with_count[1])
        summed_weights += word_with_count[1]

    if len(possible_words) == 0:
        return get_word_from_n_gram_model(cur_n-1, cur_title, language_models)
    else:
        scaled_weights = []
        cumsum = 0
        for weight in weights:
            cumsum += weight
            #scaled_weights.append(cumsum/summed_weights)
            scaled_weights.append(weight/summed_weights)

        next_word = np.random.choice(possible_words, 1, scaled_weights)[0]
        return next_word

# %%
# 3.3 If you didn't just copy what nltk's lm.generate does: compare the
#     outputs
def generate_ntlk_text(model, length, text_seed: str='<s>'):
    seed = text_seed.split()
    if text_seed == '<s>':
        length += 1
    title = []
    title.append(text_seed)
    for token in model.generate(num_words = length, text_seed=seed):
        if token == '<s>' or token == "":
            continue
        if token == '</s>':
            title.append(token)
            break
        title.append(token)
    return ' '.join(title)

def compare_text_generation(n: int, length: int, language_models:dict, seed: str='<s>') -> str:
    print('Custom Text Generation: ')
    print(generate(n, length, language_models, seed))
    print()
    print('NLTK Text Generation: ')
    print(generate_ntlk_text(language_models[n], length, text_seed=seed))

compare_text_generation(3, 10, language_models)