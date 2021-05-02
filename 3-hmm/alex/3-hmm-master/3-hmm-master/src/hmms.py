#!/usr/bin/env python3

# %%

import librosa
import numpy as np
from hmmlearn import hmm

# be reproducible...
np.random.seed(1337)

# ---%<------------------------------------------------------------------------
# Part 1: Basics

# version 1.0.10 has 10 digits, spoken 50 times by 6 speakers
digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
nr = 50
speakers = list(['george', 'jackson', 'lucas', 'nicolas', 'theo', 'yweweler'])

# %%
def load_fts(digit: int, spk: str, n: int):
    # load sounds file, compute MFCC; eg. n_mfcc=13
    pass

# load data files and extract features

# %% 

# implement a 6-fold cross-validation (x/v) loop so that each speaker acts as
# test speaker while the others are used for training

# allocate and initialize the HMMs, one for each digit; set a linear topology
# choose and a meaningful number of states
# note: you may find that one or more HMMs are performing particularly bad;
# what could be the reason and how to mitigate that?

# train the HMMs using the fit method; data needs to be concatenated,
# see https://github.com/hmmlearn/hmmlearn/blob/38b3cece4a6297e978a204099ae6a0a99555ec01/lib/hmmlearn/base.py#L439

# evaluate the trained models on the test speaker; how do you decide which word
# was spoken?

# compute and display the confusion matrix
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

# %%
# display the overall confusion matrix


# ---%<------------------------------------------------------------------------
# Part 2: Decoding

# generate test sequences; retain both digits (for later evaluation) and
# features (for actual decoding)

# combine the (previously trained) per-digit HMMs into one large meta HMM; make
# sure to change the transition probabilities to allow transitions from one
# digit to any other

# use the `decode` function to get the most likely state sequence for the test
# sequences; re-map that to a sequence of digits

# use jiwer.wer to compute the word error rate between reference and decoded
# digit sequence

# compute overall WER (ie. over the cross-validation)

# ---%<------------------------------------------------------------------------
# Optional: Decoding

