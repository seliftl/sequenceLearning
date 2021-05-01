#!/usr/bin/env python3

# %%

import librosa
import numpy as np
import hmmlearn.hmm as hmm

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
    filename = str(digit) + '_' + spk + '_' + str(n)
    y, sr = librosa.load('../res/'+filename+'.wav', sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=int(0.010*sr), n_fft=int(0.025*sr), n_mfcc=13)
    return mfcc

def load():
    mfccs = {}
    for speaker in speakers:
        digit_list = [] 
        for digit in digits:
            rec_list = []
            for i in range(0, 50):
                mfcc = load_fts(digit, speaker, i)
                rec_list.append(mfcc)
            digit_list.append(rec_list)
        mfccs[speaker] = digit_list
    return mfccs

mfccs = load()
# %%
# Sample output
mfccs['george'][0][0]
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

def prepare_train_data(train_mfccs: dict):
    train_data = {}
    for i in range(0,len(digits)):
        digit_recs = []
        for speaker_data in train_mfccs.keys():
            for j in range(0, nr):
                digit_recs.append(train_mfccs[speaker_data][i][j])
        flatlist = [item for sublist in digit_recs for item in sublist]
        y = np.concatenate(flatlist, axis=0)
        train_data[i] = y.reshape(-1, 1)
    return train_data

def train_hmms(train_mfccs: dict):
    trained_hmms={}
    for label in train_mfccs.keys():
        model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=10)
        model.fit(train_mfccs[label])  
        trained_hmms[label] = model
    return trained_hmms

def test_hmms(trained_hmms:dict, test_mfccs: list):
    for i in range(0, len(test_mfccs)):
        flatlist = [item for sublist in test_mfccs[i] for item in sublist]
        y = np.concatenate(flatlist, axis=0)
        features = y.reshape(-1, 1)
        scoreList = {}
        for model_label in trained_hmms.keys():
            model = trained_hmms[model_label]
            score = model.score(features)
            scoreList[model_label] = score
        predict = max(scoreList, key=scoreList.get)
        print("True label ", i, ": Predicted Label: ", predict)

def cross_validation(mfccs: dict):    
    for speaker in speakers: 
        train_mfccs = mfccs.copy()
        del train_mfccs[speaker]
        train_data = prepare_train_data(train_mfccs)
        trained_hmms = train_hmms(train_data)
        test_hmms(trained_hmms, mfccs[speaker])
cross_validation(mfccs)


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

