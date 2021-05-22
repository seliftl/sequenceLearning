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
from sub.models import LSTMClassifier
# %%
# Part 1
# ---%<------------------------------------------------------------------------
def prepare_data():    
    theses_df = pd.read_csv('../res/theses.tsv',header=None, sep='\t')

    # remove diploma theses
    theses_df = theses_df[theses_df[2] != 'Diplom'] 
    del theses_df[0]
    theses_df.rename(columns={1: 'ort', 2: 'art',  3: 'text'}, inplace=True)
    theses_df = theses_df[['text', 'ort', 'art']]
    theses_df.to_csv('theses_prepared.csv', encoding='utf-8', index=False)

titles = prepare_data()
# %%
def tokenize_data():
  tokenize = lambda x: x.split()
  TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=200)
  LABEL = data.LabelField(dtype = torch.float,batch_first=True)
  fields = [('text',TEXT), (None, None),('art', LABEL)]

  #loading custom dataset
  training_data=data.TabularDataset(path = 'theses_prepared.csv',format = 'csv',fields = fields,skip_header = True)
  return training_data, TEXT, LABEL

training_data, TEXT, LABEL = tokenize_data()
#print preprocessed text
print(vars(training_data.examples[0]))
# %%
def build_vocab():
  SEED = 2019
  train_data, test_data  = training_data.split(split_ratio=0.8, random_state = random.seed(SEED))
  TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
  LABEL.build_vocab(train_data)
  return train_data, test_data, TEXT, LABEL

train_data, test_data, TEXT, LABEL = build_vocab()

print("Size of TEXT vocabulary:",len(TEXT.vocab)) #No. of unique tokens in text
print("Size of LABEL vocabulary:",len(LABEL.vocab)) #No. of unique tokens in label
print(TEXT.vocab.freqs.most_common(10))  #Commonly used words
# print(TEXT.vocab.stoi)  #Word dictionary
# %%
def split_train_data(train_data):
  train_data, valid_data = train_data.split() # Further splitting of training_data to create new training_data & validation_data
  train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=32, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)
  word_embeddings = TEXT.vocab.vectors
  vocab_size = len(TEXT.vocab)
  return train_data, valid_data, train_iter, valid_iter, test_iter, word_embeddings, vocab_size

train_data, valid_data, train_iter, valid_iter, test_iter, word_embeddings, vocab_size = split_train_data(train_data)
# %%
class RNN(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
		super(RNN, self).__init__()

		self.batch_size = batch_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		
		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
		self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
		self.rnn = nn.RNN(embedding_length, hidden_size, num_layers=2, bidirectional=True)
		self.label = nn.Linear(4*hidden_size, output_size)
	
	def forward(self, input_sentences, batch_size=None):
		input = self.word_embeddings(input_sentences)
		input = input.permute(1, 0, 2)
		if batch_size is None:
			h_0 = Variable(torch.zeros(4, self.batch_size, self.hidden_size)) # 4 = num_layers*num_directions
		else:
			h_0 =  Variable(torch.zeros(4, batch_size, self.hidden_size))
		output, h_n = self.rnn(input, h_0)
		h_n = h_n.permute(1, 0, 2) # h_n.size() = (batch_size, 4, hidden_size)
		h_n = h_n.contiguous().view(h_n.size()[0], h_n.size()[1]*h_n.size()[2])
		logits = self.label(h_n) # logits.size() = (batch_size, output_size)
		
		return logits
# %%
def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model, train_iter, epoch):
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

def eval_model(model, val_iter):
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
            pred_vec = torch.max(prediction, 1)[1].view(target.size()).data

            num_false_positives = (pred_vec - target == 1).sum()
            num_false_negatives = (pred_vec - target == -1).sum()
            num_corrects = (pred_vec == target.data).sum()

            acc = 100.0 * num_corrects/len(batch)
            precision = num_corrects/(num_corrects + num_false_positives)
            recall = num_corrects/(num_corrects + num_false_negatives)
            f1 = (2 * precision * recall) / (precision + recall)

            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()
            total_epoch_pres += precision.item()
            total_epoch_rec += recall.item()
            total_epoch_f1 += f1.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter), total_epoch_pres/len(val_iter), total_epoch_rec/len(val_iter), total_epoch_f1/len(val_iter)
	

learning_rate = 2e-5
batch_size = 32
output_size = 2
hidden_size = 256
embedding_length = 300

model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
loss_fn = F.cross_entropy

for epoch in range(10):
    train_loss, train_acc = train_model(model, train_iter, epoch)
    val_loss, val_acc, val_pres, val_rec, val_f1 = eval_model(model, valid_iter)
    
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%, Val. Pres: {val_pres:.2f}%, Val. Rec: {val_rec:.2f}%, Val. F1: {val_f1:.2f}%')
    
test_loss, test_acc, test_pres, test_rec, test_f1 = eval_model(model, test_iter)
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%, Test Pres: {test_pres:.2f}%, Test Rec: {test_rec:.2f}%, Test F1: {test_f1:.2f}%')
# %%
