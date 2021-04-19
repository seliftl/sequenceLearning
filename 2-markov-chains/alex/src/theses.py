#!/usr/bin/env python3
#%%
# we will be using nltk.lm and numpy
import numpy as np
import nltk
import io
import re
from nltk import ngrams
from nltk.lm import Vocabulary
from nltk.lm.models import Laplace

# 0. Before you get started, make sure to download the `theses.txt` data set.
# %%
def load_data() -> list:
    file = io.open("../res/theses.txt", mode="r", encoding="utf-8")
    theses = file.readlines()
    theses = [x.strip() for x in theses] 
    return theses
theses = load_data()

# 1. Spend some time on pre-processing. How would you handle hyphenated words
#    and abbreviations/acronyms?

# %%
def prepare_data(theses: list) -> list:
    preprocessed_theses = []
    for title in theses:        
        # Remove punctuations and numbers except dashes in-between words
        # title = re.sub('[^a-zäöüA-ZÄÖÜ-]', ' ', title)
        title = re.sub('[- ]', ' ', title)
        title = re.sub('[?!.:,]', ' ', title)
        # Remove multiple spaces
        title = re.sub(r'\s+', ' ', title)
        preprocessed_theses.append(title.lower())
    return preprocessed_theses
preprocessed_data = prepare_data(theses)
print(preprocessed_data)


# 2. Train n-gram models with n = [1, ..., 5]. What about <s> and </s>?
# %%
# Tokenization
def tokenize_text(text):  
    tokens = nltk.word_tokenize(text)
    tokens.insert(0, '<s>')
    tokens.append('</s>')
    return tokens


titles_as_tokens = [tokenize_text(title) for title in theses]
print('Sample output tokenization')
print(titles_as_tokens[0])

# %%
# N-Gramm Generation
def get_dict_of_ngrams(training_data, n_grams=0):
    if n_grams == 0:
        n_grams = {1:[], 2:[], 3:[], 4:[], 5:[]}

    for title in training_data:
        for i in range(1,6):
          n_grams[i].append(list(ngrams(title, i)))
      
    return n_grams

dict_of_ngrams = get_dict_of_ngrams(titles_as_tokens)
print('Sample output ngrams')
print(1, dict_of_ngrams[1][0], '\n')
print(2, dict_of_ngrams[2][0], '\n')
print(3, dict_of_ngrams[3][0], '\n')
print(4, dict_of_ngrams[4][0], '\n')
print(5, dict_of_ngrams[5][0], '\n')

# %%
# Vocab Generation
vocab = Vocabulary(list(nltk.flatten(titles_as_tokens)))

# %%
# Training of Models
def train_model(n, training_data, vocab):
    lm = Laplace(n)
    lm.fit(training_data, vocab)
    return lm

def get_models(dict_of_ngrams, vocab):
    models = {}
    for i in range(1,6):
        models[i] = train_model(i, dict_of_ngrams[i], vocab)
        print('Model for ', i, '-grams trained')
    return models

language_models = get_models(dict_of_ngrams, vocab) 

# 3. Write a generator that provides thesis titles of desired length. Please
#    do not use the available `lm.generate` method but write your own.
#    nb: If you fix the seed in numpy.random.choice, you get reproducible 
#        results.
# 3.1 How can you incorporate seed words?
# 3.2 How do you handle </s> tokens (w.r.t. the desired length?)

# 3.3 If you didn't just copy what nltk's lm.generate does: compare the
#     outputs
# %%
def generate(n: int, length: int, language_models:dict, seed: str='<s>') -> str:
    title = [seed]

    while len(title) < length:
        last_n_minus_one_words = tuple(title[-(n-1):])
        res = language_models[n].counts[last_n_minus_one_words].items()

        possible_words = []
        weights = []
        summed_weights = 0
        for word_with_count in res:
            possible_words.append(word_with_count[0])
            weights.append(word_with_count[1])
            summed_weights += word_with_count[1]

        scaled_weights = []
        cumsum = 0
        for weight in weights:
            cumsum += weight
            #scaled_weights.append(cumsum/summed_weights)
            scaled_weights.append(weight/summed_weights)

        next_word = np.random.choice(possible_words, 1, scaled_weights)
        title.append(next_word[0])
    
    return title

# %%
#res = language_models[2].counts[('<s>',)].items()
print(generate(3, 10, language_models))

# %%
# Keine Worte zuvor
res = language_models[2].counts[2].items()
print(res)
for i, ele in enumerate(res):
    if i > 5:
        break
    print(ele)

# %%
res = language_models[1].counts[1].items()
print(res)
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

# %%
print(np.random.choice(possible_words, 1, scaled_weights))

# %%

res = language_models[2].counts[('<s>',)].items()
print(res)
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

# %%
print(np.random.choice(possible_words, 1, scaled_weights))

# %%
res = language_models[3].counts[('<s>', 'Konzeption')].items()
print(res)
for element in res:
    print(element)

# %%
res = language_models[2].counts[['<s>']]

print(res)
for element in res:
    print(element)
    print(res[element])

# %%
following_words_dict = language_models[2].counts[['<s>']]
possible_words = []
weights = []
summed_weights = 0
for word in following_words_dict:
    possible_words.append(word)
    cur_weight = following_words_dict[word]
    weights.append(cur_weight)
    summed_weights += cur_weight

scaled_weights = []
for weight in weights:
    scaled_weights.append(weight/summed_weights)


# %%
print(np.random.choice(possible_words, 1, scaled_weights))
# %%
