#!/usr/bin/env python3
#%%
# we will be using nltk.lm and numpy
import numpy as np
import nltk
import io
import re
from nltk import ngrams
#%%
# 0. Before you get started, make sure to download the `theses.txt` data set.
def load_data() -> list:
    file = io.open("../res/theses.txt", mode="r", encoding="utf-8")
    theses = file.readlines()
    theses = [x.strip() for x in theses] 
    return theses
theses = load_data()
#%%
# 1. Spend some time on pre-processing. How would you handle hyphenated words
#    and abbreviations/acronyms?
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

#%%
# 2. Train n-gram models with n = [1, ..., 5]. What about <s> and </s>?
def train(theses: list, n: int):
    ngrams = {}
    for title in theses:
        tokens = nltk.word_tokenize(title)
        tokens.insert(0, '<s>')
        tokens.append('</s>')
        for i in range(len(tokens)-n):
            seq = ' '.join(tokens[i:i+n])
            if  seq not in ngrams.keys():
                ngrams[seq] = []
            ngrams[seq].append(tokens[i+n])
    return ngrams
ngrams = train(preprocessed_data, 2)
#%%
# 3. Write a generator that provides thesis titles of desired length. Please
#    do not use the available `lm.generate` method but write your own.
#    nb: If you fix the seed in numpy.random.choice, you get reproducible 
#        results.
# 3.1 How can you incorporate seed words?
# 3.2 How do you handle </s> tokens (w.r.t. the desired length?)
def generate(n: int, length: int, ngrams: dict, seed: str = "") -> str:
    # if no seed: <s> as seed?
    seedlist = ['<s>']
    if seed == "":
        # choose one key of ngram randomly as seed that starts with <s>
        keys = [key for key, value in ngrams.items() if '<s>' in key]
        seedlist = keys[np.random.choice(len(keys))].split()
    else:
        seedlist.append(seed)
    curr_words = ' '.join(seedlist)
    output = curr_words
    for i in range(length-n):
        if curr_words not in ngrams.keys():
            break
        possible_words = ngrams[curr_words]
        if i < (length-n-1):
            # try to eliminate </s> before end
            reduced_words = [word for word in possible_words if not '</s>' in word]
            if len(reduced_words)!=0:
                possible_words = reduced_words
        next_word = possible_words[np.random.choice(len(possible_words))]
        output += ' ' + next_word
        tokenized_output = nltk.word_tokenize(output)
        curr_words = ' '.join(tokenized_output[len(tokenized_output)-n:len(tokenized_output)])

    return output
generate(2, 15, ngrams)
#%%
# 3.3 If you didn't just copy what nltk's lm.generate does: compare the
#     outputs
def compare(n: int):
    # call custom generate
    # call lm.generate
    pass