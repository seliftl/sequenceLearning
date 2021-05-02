#!/usr/bin/env python3

# %%

import librosa
import numpy as np
from hmmlearn import hmm
import os

# be reproducible...
np.random.seed(1337)

# ---%<------------------------------------------------------------------------
# Part 1: Basics

# version 1.0.10 has 10 digits, spoken 50 times by 6 speakers
digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
nr = 50
speakers = list(['george', 'jackson', 'lucas', 'nicolas', 'theo', 'yweweler'])

# %%
# load sounds file, compute MFCC; eg. n_mfcc=13
def load_fts(digit: int, spk: str, n: int):
    directory = os.path.dirname(__file__) + '/../res/recordings'
    file_name_start = str(digit) + '_' + spk
    stored_mfccs = []

    for filename in os.listdir(directory):
        if filename.startswith(file_name_start) and filename.endswith('.wav'):
            samples, sample_rate = librosa.load(os.path.join(directory, filename))
            mfccs = librosa.feature.mfcc(samples, sample_rate, n_mfcc=n)
            mfccs = np.transpose(mfccs)
            stored_mfccs.append(mfccs)

    return stored_mfccs

def load_speaker_fts_as_dict(spk: str):
    dict_of_speaker = {}
    for i in digits:
        dict_of_speaker[i] = load_fts(i, spk, 13)
    return dict_of_speaker

# %%
# load data files and extract features
speaker_dict = {}
for speaker in speakers:
    speaker_dict[speaker] = load_speaker_fts_as_dict(speaker)



# %% 

# implement a 6-fold cross-validation (x/v) loop so that each speaker acts as
# test speaker while the others are used for training

#for test_speaker in speakers:
test_speaker = speakers[0]
training_speakers = [speaker for speaker in speakers if speaker != test_speaker]
training_data_dict = {}

for training_speaker in training_speakers:
    for digit in digits:
        if digit not in training_data_dict:
            training_data_dict[digit] = []

        training_data_dict[digit].extend(speaker_dict[training_speaker][digit])

test_data_dict = {}
for digit in digits:
        if digit not in test_data_dict:
            test_data_dict[digit] = []
        
        test_data_dict[digit] = speaker_dict[speaker][digit]

# allocate and initialize the HMMs, one for each digit; set a linear topology
# choose and a meaningful number of states
hmms = {}
for digit in digits:
    # create models
    hmms[digit] = hmm.GaussianHMM(n_components=3, covariance_type="diag",
                    init_params="cm", params="cmt")
    hmms[digit].startprob_ = np.array([1.0, 0.0, 0.0])
    hmms[digit].transmat_ = np.array([[1.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 0.0, 1.0]])

    # prep train data
    conc_train_data = 0
    lenghts = []
    for index, train_data in enumerate(training_data_dict[digit]):
        if index == 0:
            conc_train_data = train_data
        else:
            conc_train_data = np.concatenate([conc_train_data, train_data])
        
        lenghts.append(len(train_data))

    # fit models
    hmms[digit].fit(conc_train_data, lenghts)

# %%
# allocate and initialize the HMMs, one for each digit; set a linear topology
# choose and a meaningful number of states
hmms = {}
for digit in digits:
    hmms[digit] = hmm.GaussianHMM(n_components=3, covariance_type="diag",
                     init_params="cm", params="cmt")
    hmms[digit].startprob_ = np.array([1.0, 0.0, 0.0])
    hmms[digit].transmat_ = np.array([[1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [0.0, 0.0, 1.0]])

# %%
# prep train data
conc_train_data = 0
lenghts = []
for index, train_data in enumerate(training_data_dict[0]):
    if index == 0:
        conc_train_data = train_data
    else:
        conc_train_data = np.concatenate([conc_train_data, train_data])
    
    lenghts.append(len(train_data))

test = conc_train_data

# note: you may find that one or more HMMs are performing particularly bad;
# what could be the reason and how to mitigate that?

# train the HMMs using the fit method; data needs to be concatenated,
# see https://github.com/hmmlearn/hmmlearn/blob/38b3cece4a6297e978a204099ae6a0a99555ec01/lib/hmmlearn/base.py#L439

# evaluate the trained models on the test speaker; how do you decide which word
# was spoken?

# compute and display the confusion matrix
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

# Addiditonal Resources:
# https://machinelearningmastery.com/k-fold-cross-validation/
# https://medium.com/voice-tech-podcast/single-word-speech-recognition-892c7e01f5fc
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

