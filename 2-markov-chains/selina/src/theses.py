#!/usr/bin/env python3

# we will be using nltk.lm and numpy
import numpy as np
import nltk

#%%
# 0. Before you get started, make sure to download the `theses.txt` data set.
def load_data() -> list:
    pass

#%%
# 1. Spend some time on pre-processing. How would you handle hyphenated words
#    and abbreviations/acronyms?
def prepare_data(thesis: list) -> list:
    # tolower 
    # keep hyphens
    pass

#%%
# 2. Train n-gram models with n = [1, ..., 5]. What about <s> and </s>?
def train(thesis: list, n: int):
    # use nltk.ngrams
    pass
#%%
# 3. Write a generator that provides thesis titles of desired length. Please
#    do not use the available `lm.generate` method but write your own.
#    nb: If you fix the seed in numpy.random.choice, you get reproducible 
#        results.
# 3.1 How can you incorporate seed words?
# 3.2 How do you handle </s> tokens (w.r.t. the desired length?)
def generate(n: int, seed: str = "") -> str:
    # if no seed: <s> as seed?
    pass

#%%
# 3.3 If you didn't just copy what nltk's lm.generate does: compare the
#     outputs
def compare(n: int):
    # call custom generate
    # call lm.generate
    pass