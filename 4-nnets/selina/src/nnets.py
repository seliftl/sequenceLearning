#!/usr/bin/env python3

# %%

import librosa
import numpy as np
from hmmlearn import hmm
import os
from numpy.lib.function_base import append
from pandas.core import api
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
from jiwer import wer
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
            
            test_data_dict[digit] = speaker_dict[test_speaker][digit]
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
                test = model.means_
                test2 = model.covars_
                #pred = model.predict(test_sample)
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

def generate_test_seq(num_digits: int, speaker: str):
    test_seq = []
    true_words = [] 
    lengths = []
    for i in range(0, num_digits):
        word = np.random.randint(0, 10)
        true_words.append(word)
        rec = np.random.randint(0, 50)
        mfccs = speaker_dict[speaker][word][rec]
        lengths.append(len(mfccs))
        test_seq.append(mfccs)
    return test_seq, true_words, lengths
#%%
# combine the (previously trained) per-digit HMMs into one large meta HMM; make
# sure to change the transition probabilities to allow transitions from one
# digit to any other
def generate_meta_hmm(hmms: dict):

    startprops_of_hmm = []
    transmat_of_meta_hmm = []
    means_of_meta_hmm = []
    covariances_of_meta_hmm = []

    for digit in hmms.keys():
        # Add each hmm to meta-hmm
        for added_state_index in range(3):
            # Per hmm three states will be added

            # First handle startprops of each state
            # Each first state of exisiting hmms gets start prop 0.1
            # Other two get 0
            if added_state_index == 0:
                startprops_of_hmm.append(0.1)
            else:
                startprops_of_hmm.append(0)

            # Then hanlde transition props
            transition_props_of_state = []
            for meta_hmm_target_state in range(30):
                # Per added state 30 transition props (for all 30 states) need to be set
                if added_state_index < 3:
                    # handle first two states of existing hmm
                    if meta_hmm_target_state >= digit*3 and meta_hmm_target_state <= digit*3+2:
                        # transition to own states can be set from learned transition of existing hmm
                        transition_props_of_state.append(hmms[digit].transmat_[added_state_index][meta_hmm_target_state%3])
                    else:
                        # other transitions are set to 0
                        transition_props_of_state.append(0)
                else:
                    # for the last state of exisiting hmm props for to all first states of exisinting hmms should be possible
                    if meta_hmm_target_state%3 == 0:
                        transition_props_of_state.append(0.1)
                    else:
                        transition_props_of_state.append(0)

            transmat_of_meta_hmm.append(transition_props_of_state)

            # After transition props, handle means and covariances
            means_of_meta_hmm.append(hmms[digit].means_[added_state_index])
            covariances_of_meta_hmm.append(hmms[digit].covars_[added_state_index])
    

    meta_hmm = hmm.GaussianHMM(n_components=30, covariance_type="full")

    meta_hmm.startprob_ = np.array(startprops_of_hmm)
    meta_hmm.transmat_ = np.array(transmat_of_meta_hmm)
    meta_hmm.means_ = np.array(means_of_meta_hmm)
    meta_hmm.covars_ = np.array(covariances_of_meta_hmm)
    
    return meta_hmm

# %%
# use the `decode` function to get the most likely state sequence for the test
# sequences; re-map that to a sequence of digits
def decode(meta_hmm, test_seq, lengths):
    for index, test_data in enumerate(test_seq):
        if index == 0:
            conc_test_data = test_data
        else:
            conc_test_data = np.concatenate([conc_test_data, test_data])
    logprob, state = meta_hmm.decode(conc_test_data, lengths, algorithm="viterbi")
    return state

def map_state_to_digit(states, lengths):
    hyp_words = []
    for i in range(0, len(lengths)):
        state = states[:lengths[i]]
        states = states[lengths[i]:]
        digits = [int(x / 3) for x in state]
        hyp_words.append(max(digits, key = digits.count))
    return hyp_words

#%%
# use jiwer.wer to compute the word error rate between reference and decoded
# digit sequence
# compute overall WER (ie. over the cross-validation)
def cross_valid_meta(num_digits: int):
    errors = []
    for test_speaker in speakers:
        print(test_speaker)
        training_data_dict, test_data_dict = prepare_data(test_speaker)
        hmms = train_hmms(training_data_dict)
        meta_hmm = generate_meta_hmm(hmms)
        for i in range(0, 10):
            test_seq, true_words, lengths = generate_test_seq(num_digits, test_speaker)
            states = decode(meta_hmm, test_seq, lengths)
            hyp_words = map_state_to_digit(states, lengths)
            error = wer(' '.join(str(true_words)), ' '.join(str(hyp_words)))
            errors.append(error)
    print(errors)
    print(np.mean(errors))

cross_valid_meta(3)
# %%
