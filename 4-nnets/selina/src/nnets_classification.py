#!/usr/bin/env python3

# %%
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchtext.legacy import data
from torchtext.vocab import Vectors, GloVe 
from model.models import LSTM, BiLSTM, GRU, RNN
from sklearn.model_selection import KFold
# %%
# rearrange data with pandas
def prepare_data():    
    theses_df = pd.read_csv('../res/theses.tsv',header=None, sep='\t')

    # remove diploma theses
    theses_df = theses_df[theses_df[2] != 'Diplom'] 
    del theses_df[0]

    # reorder columns
    theses_df.rename(columns={1: 'ort', 2: 'art',  3: 'text'}, inplace=True)
    theses_df = theses_df[['text', 'ort', 'art']]
    return theses_df

# %%
# build and label train and test vocabulary 
def build_vocab(classification_type: str):
    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=200)
    LABEL = data.LabelField(sequential=False, fix_length = 2, dtype = torch.float, batch_first=True)
    
    # define fields according to classification type
    if classification_type == 'art':
        fields = [('text',TEXT), (None, None),('art', LABEL)]
    else:
        fields = [('text',TEXT), ('ort', LABEL),(None, None)]
    train_data, test_data = data.TabularDataset.splits(path='data/', train='theses_train.csv', validation=None, test='theses_test.csv', format='csv', fields = fields)
    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train_data)
    return train_data, test_data, TEXT, LABEL

# %%
# get split iterators
def split_train_data(train_data, test_data, TEXT):
  train_iter, test_iter = data.BucketIterator.splits((train_data, test_data), batch_size=32, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)
  word_embeddings = TEXT.vocab.vectors
  vocab_size = len(TEXT.vocab)
  return train_data, train_iter, test_iter, word_embeddings, vocab_size

# %%
# train and evaluate model
def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model, train_iter, epoch, loss_fn):
    total_epoch_loss = 0
    total_epoch_acc = 0
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.art
        target = torch.autograd.Variable(target).long()
        if (text.size()[0] is not 32):# One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        prediction = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        
        if steps % 100 == 0:
            print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(model, val_iter, loss_fn):
    total_epoch_loss = 0
    total_epoch_acc = 0
    total_epoch_pres = 0
    total_epoch_rec = 0
    total_epoch_f1 = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            if (text.size()[0] is not 32):
                continue
            target = batch.art
            target = torch.autograd.Variable(target).long()
            prediction = model(text)
            loss = loss_fn(prediction, target)

            # get prediction vector of zeros and ones for batch
            pred_vec = torch.max(prediction, 1)[1].view(target.size()).data

            # get numbers of wrong/correct classifications
            num_false_positives = (pred_vec - target == 1).sum()
            num_false_negatives = (pred_vec - target == -1).sum()
            num_corrects = (pred_vec == target.data).sum()

            # calculate metrics
            acc = 100.0 * num_corrects/len(batch)
            precision = 100.0 * num_corrects/(num_corrects + num_false_positives)
            recall = 100.0 * num_corrects/(num_corrects + num_false_negatives)
            f1 =  (2 * precision * recall) / (precision + recall)

            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()
            total_epoch_pres += precision.item()
            total_epoch_rec += recall.item()
            total_epoch_f1 += f1.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter), total_epoch_pres/len(val_iter), total_epoch_rec/len(val_iter), total_epoch_f1/len(val_iter)
	
def run_model(model, vocab_size, word_embeddings, train_iter, test_iter):
    batch_size = 32
    output_size = 2
    hidden_size = 256
    embedding_length = 300

    model = model(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    loss_fn = F.cross_entropy

    for epoch in range(10):
        train_loss, train_acc = train_model(model, train_iter, epoch, loss_fn)    
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%')
        
    test_loss, test_acc, test_pres, test_rec, test_f1 = eval_model(model, test_iter, loss_fn)
    print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%, Test Pres: {test_pres:.2f}%, Test Rec: {test_rec:.2f}%, Test F1: {test_f1:.2f}%')
    return test_acc, test_pres, test_rec, test_f1
# %%
def cross_valid(classification_type: str, model):
    accuracy = []
    precision = []
    recall = []
    f1 = []

    theses_df = prepare_data()
    cv = KFold(n_splits=5, shuffle=False)
    for train_index, test_index in cv.split(theses_df):

        X_train, X_test= theses_df.iloc[train_index], theses_df.iloc[test_index]
        # save train and test data chunks
        X_train.to_csv('data/theses_train.csv', encoding='utf-8', index=False, header=False)
        X_test.to_csv('data/theses_test.csv', encoding='utf-8', index=False, header=False)

        train_data, test_data, TEXT, LABEL = build_vocab(classification_type)
        train_data, train_iter, test_iter, word_embeddings, vocab_size = split_train_data(train_data, test_data, TEXT)
        test_acc, test_pres, test_rec, test_f1 = run_model(model, vocab_size, word_embeddings, train_iter, test_iter)
        accuracy.append(test_acc)
        precision.append(test_pres)
        recall.append(test_rec)
        f1.append(test_f1)
    print('Overall Accuracy:', np.mean(accuracy))
    print('Overall Precision:', np.mean(precision))
    print('Overall Recall:', np.mean(recall))
    print('Overall F1:', np.mean(f1))

# insert preferred model and classification type
# as training takes quite a while, an overview of the model scores is provided in "model_scores.pdf"
rnn = RNN
gru = GRU
lstm = LSTM
bilstm = BiLSTM
cross_valid('art', gru)
# %%
