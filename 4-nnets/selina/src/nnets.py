#!/usr/bin/env python3

# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.autograd import Variable
import torch.functional as F
import torch.nn.functional as F

# %%
# Part 1
# ---%<------------------------------------------------------------------------
def read_data():    
    theses_df = pd.read_csv('../res/theses.tsv',header=None, sep='\t')

    # remove in final solution
    theses_df = theses_df.head(100)    
    titles = theses_df[3].tolist()
    return titles

def tokenize_corpus(titles):
    titles = [x.lower() for x in titles]
    tokenized_titles = [x.split() for x in titles]
    return tokenized_titles

titles = read_data()
tokenized_titles = tokenize_corpus(titles)

# %%
def create_vocab():
    vocabulary = []
    for sentence in tokenized_titles:
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)

    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

    vocabulary_size = len(vocabulary)
    return word2idx, idx2word, vocabulary_size

word2idx, idx2word, vocabulary_size = create_vocab()
# %%
def create_context(window_size: int):
    idx_pairs = []
    # for each sentence
    for sentence in tokenized_titles:
        indices = [word2idx[word] for word in sentence]
        # for each word, threated as center word
        for center_word_pos in range(len(indices)):
            # for each window position
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w
                # make soure not jump out sentence
                if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                    continue
                context_word_idx = indices[context_word_pos]
                idx_pairs.append((indices[center_word_pos], context_word_idx))

    idx_pairs = np.array(idx_pairs) # it will be useful to have this as numpy array
    return idx_pairs

idx_pairs = create_context(4)

# %%  
def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x

def train_skipgram():
    embedding_dims = 5
    W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
    W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
    num_epochs = 100
    learning_rate = 0.001

    for epo in range(num_epochs):
        loss_val = 0
        for data, target in idx_pairs:
            x = Variable(get_input_layer(data)).float()
            y_true = Variable(torch.from_numpy(np.array([target])).long())

            z1 = torch.matmul(W1, x)
            z2 = torch.matmul(W2, z1)
        
            log_softmax = F.log_softmax(z2, dim=0)

            loss = F.nll_loss(log_softmax.view(1,-1), y_true)
            loss_val += loss.item()
            loss.backward()
            W1.data -= learning_rate * W1.grad.data
            W2.data -= learning_rate * W2.grad.data

            W1.grad.data.zero_()
            W2.grad.data.zero_()
        if epo % 10 == 0:    
            print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')
    return W1, W2

# %%
W1, W2 = train_skipgram()
# %%
def similarity(v,u):
  return torch.dot(v,u)/(torch.norm(v)*torch.norm(u))

def find_most_similar_word(word):
    similarities = {}
    w1v = torch.matmul(W1,get_input_layer(word2idx[word]))
    for i in range(len(idx2word)):
        compared_word = idx2word[i]
        w2v = torch.matmul(W1,get_input_layer(word2idx[compared_word]))
        computed_similarity = similarity(w1v, w2v)
        similarities[compared_word] = computed_similarity
    most_similar_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    print(most_similar_words[:3])

find_most_similar_word('konzeption')
# %%
# Part 2
# ---%<------------------------------------------------------------------------
idx_pairs[:100]
# %%
