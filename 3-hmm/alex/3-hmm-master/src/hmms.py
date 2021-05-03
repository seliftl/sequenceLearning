#!/usr/bin/env python3

# %%

import librosa
import numpy as np
from hmmlearn import hmm
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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
    directory = os.path.dirname(__file__) + '/../res'
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
# note: you may find that one or more HMMs are performing particularly bad;
# what could be the reason and how to mitigate that?

# train the HMMs using the fit method; data needs to be concatenated,
# see https://github.com/hmmlearn/hmmlearn/blob/38b3cece4a6297e978a204099ae6a0a99555ec01/lib/hmmlearn/base.py#L439

# compute and display the confusion matrix
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

# Addiditonal Resources:
# https://machinelearningmastery.com/k-fold-cross-validation/
# https://medium.com/voice-tech-podcast/single-word-speech-recognition-892c7e01f5fc

def prepare_data(test_speaker: str):
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
    return training_data_dict, test_data_dict

# allocate and initialize the HMMs, one for each digit; set a linear topology
# choose and a meaningful number of states
def train_hmms(training_data_dict: dict):
    hmms = {}
    for digit in digits:
        # create models
        hmms[digit] = hmm.GaussianHMM(n_components=3, covariance_type="diag",
                        init_params="cm", params="cmt")
        hmms[digit].startprob_ = np.array([1.0, 0.0, 0.0])
        hmms[digit].transmat_ = np.array([[0.5, 0.5, 0.0],
                                        [0.0, 0.5, 0.5],
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
    return hmms

# evaluate the trained models on the test speaker; how do you decide which word
# was spoken?
def test_hmms(hmms: dict, test_data_dict: dict):
    predict_res = {}
    for digit in test_data_dict.keys(): 
        pred_list = []
        for test_sample in test_data_dict[digit]:
            scoreList = {}            
            for model_digit in hmms.keys():            
                model = hmms[model_digit]
                score = model.score(test_sample)
                scoreList[model_digit] = score
            predict = max(scoreList, key=scoreList.get)
            pred_list.append(predict)
        print("True label ", digit, ": Predicted Label: ", str(pred_list))
        predict_res[digit]=pred_list
    return predict_res

# implement a 6-fold cross-validation (x/v) loop so that each speaker acts as
# test speaker while the others are used for training
def cross_valid():
    speaker_pred = {}
    for test_speaker in speakers:
        print('Test Speaker:', test_speaker)
        training_data_dict, test_data_dict = prepare_data(test_speaker)
        hmms = train_hmms(training_data_dict)
        pred_res = test_hmms(hmms, test_data_dict)        
        speaker_pred[test_speaker] = pred_res
    return speaker_pred
    
# %%
speaker_pred = cross_valid()

#%%
# display the overall confusion matrix
def display_confusion_matrix(speaker_pred):
    for speaker in speaker_pred.keys():  
        y_true = []
        y_pred = []
        for digit in speaker_pred[speaker]:
            for i in range(0, len(speaker_pred[speaker][digit])):
                y_true.append(digit)
                y_pred.append(speaker_pred[speaker][digit][i])
        cm = confusion_matrix(y_true, y_pred)
        plt.matshow(cm, cmap='binary')
display_confusion_matrix(speaker_pred)
#%%
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
