# %%
import torch
from torch.autograd import Variable
import torch.nn as nn
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
class GRU(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
		super(GRU, self).__init__()

		self.batch_size = batch_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		
		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
		self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
		self.rnn = nn.GRU(embedding_length, hidden_size, num_layers=2, bidirectional=True)
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
class LSTM(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
		super(LSTM, self).__init__()

		self.batch_size = batch_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		
		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
		self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
		self.lstm = nn.LSTM(embedding_length, hidden_size, bidirectional = False)
		self.art = nn.Linear(hidden_size, output_size)
		
	def forward(self, input_sentence, batch_size=None):
	
		''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
		input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
		input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)
		if self.bidirectional == True:
			hidden_state_num = 2
		else: 
			hidden_state_num = 1
		if batch_size is None:
			h_0 = Variable(torch.zeros(hidden_state_num, self.batch_size, self.hidden_size)) # Initial hidden state of the LSTM
			c_0 = Variable(torch.zeros(hidden_state_num, self.batch_size, self.hidden_size)) # Initial cell state of the LSTM
		else:
			h_0 = Variable(torch.zeros(hidden_state_num, batch_size, self.hidden_size))
			c_0 = Variable(torch.zeros(hidden_state_num, batch_size, self.hidden_size))
		output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
		final_output = self.art(final_hidden_state[-1]) # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)
		
		return final_output
# %%
class BiLSTM(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
		super(BiLSTM, self).__init__()

		self.batch_size = batch_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		
		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
		self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
		self.lstm = nn.LSTM(embedding_length, hidden_size, bidirectional = True)
		self.art = nn.Linear(hidden_size, output_size)
		
	def forward(self, input_sentence, batch_size=None):
	
		''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
		input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
		input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)
		if self.bidirectional == True:
			hidden_state_num = 2
		else: 
			hidden_state_num = 1
		if batch_size is None:
			h_0 = Variable(torch.zeros(hidden_state_num, self.batch_size, self.hidden_size)) # Initial hidden state of the LSTM
			c_0 = Variable(torch.zeros(hidden_state_num, self.batch_size, self.hidden_size)) # Initial cell state of the LSTM
		else:
			h_0 = Variable(torch.zeros(hidden_state_num, batch_size, self.hidden_size))
			c_0 = Variable(torch.zeros(hidden_state_num, batch_size, self.hidden_size))
		output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
		final_output = self.art(final_hidden_state[-1]) # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)
		
		return final_output