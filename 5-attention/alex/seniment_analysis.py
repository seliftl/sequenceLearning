# %%
import numpy as np
import torchtext
from torch import nn
import torch
import pandas as pd
import string
import nltk
import re
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
import torch.nn.functional as F

# %%
# The first time you run this will download a ~823MB file
glove = torchtext.vocab.GloVe(name="6B", # trained on Wikipedia 2014 corpus
                              dim=50)   # embedding size = 100
# %%
nltk.download('averaged_perceptron_tagger')
#%%
def load_phrases_data(path='res/train.tsv', remove_outlier=True):
    df = pd.read_csv(path, sep='\t',
                     names=['PhraseId', 'SentenceId', 'Phrase', 'Sentiment'], skiprows=1)
    
    #tagger = ht.HanoverTagger('morphmodel_ger.pgz')
    def process_title(title):
        remove_pun = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        remove_digits = str.maketrans(string.digits, ' '*len(string.digits))
        title = title.translate(remove_digits)
        title = title.translate(remove_pun)
        title = re.sub(' {2,}', ' ', title)
        title = ' '.join([lemma for lemma, _ in nltk.pos_tag(title.split(' '))])
        return title.lower()

    df['Phrase'] = df['Phrase'].apply(process_title)
    df['Length'] = df['Phrase'].apply(lambda x: len(x.split()))
    if remove_outlier:
        df = df[df['Length'].between(4, 20)]
    vocab = df['Phrase'].str.split(expand=True).stack().value_counts().index.values
    return df, vocab

# %%
df, vocab = load_phrases_data()

# %%
class MyGRU(nn.Module):
    def __init__(self,
                input_size,
                embedding_size,
                hidden_size,
                output_size, # number of classes
                num_layers=1,
                bidirectional=False,
                vectors=None):

        super(MyGRU, self).__init__()
        self.input_size = input_size # vocabulary size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        if vectors is not None:
            # nn.Embedding can also be used with your own embeddings
            # hint: if you want to do so, you need to adapt the Dataloader
            self.embedding = nn.Embedding(
                            num_embeddings=len(vectors),
                            embedding_dim=embedding_size).from_pretrained(
                            torch.cat(list(vectors)).reshape(-1, embedding_size))
        else:
        # embedding layer as input
            self.embedding = nn.Embedding(num_embeddings=input_size,
                                          embedding_dim=embedding_size)
        self.rnn = nn.GRU(
                        input_size=embedding_size,
                        hidden_size=hidden_size,
                        num_layers=self.num_layers,
                        dropout=0.2,
                        bidirectional=bidirectional,
        )

        self.num_directions = 2 if bidirectional else 1
        # output layer (fully connected)
        fc_size = self.hidden_size * self.num_directions
        self.fc = nn.Linear(fc_size, output_size)

    def forward(self, x, h_n=None):
        if h_n is None:
            h_0 = self.init_hidden(x.size(1))
        else:
            h_0 = h_n

        embed = self.embedding(x).view(1,1,-1)
        #padded_seq = pack_padded_sequence(embed, lengths)
        # state will have dimensions of num_layers x batch_size x hidden_size
        out, h_n = self.rnn(embed, h_0)
        # out_unpacked, lens_unpacked = pad_packed_sequence(out_packed)
        # output.view(seq_len, atch, num_directions, hidden_size) for unpacked sequence
        # h_n.view(num_layers, num_directions, batch, hidden_size) # addressable per layer
        if self.num_directions == 2:
            h_forward_backward = h_n.view(2, 2, x.size(1), -1)[-1]
            h_forward_backward = torch.cat([h_forward_backward[0], h_forward_backward[0]], 1)
            logits = self.fc(h_forward_backward) #h only hidden state at last layer, if bidrect out[-1 contains the concatenated hidden state]
        else:
            logits = self.fc(h_n[-1]) #h only hidden state at last layer, if bidrect out[-1 contains the concatenated hidden state]
        # dont use batch first here, seq_len must be first dimension

        return logits, h_n # only hidden state for the last layer is needed for loss calculation

    def init_hidden(self, batch_size=1):
        # https://discuss.pytorch.org/t/lstm-hidden-state-changing-dimensions-error/23359
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        h_dim_0 = self.num_layers * self.num_directions
        hidden = torch.zeros(h_dim_0, batch_size, self.hidden_size, device=device)
        return hidden

class AttnDecoder(nn.Module):
    def __init__(self, input_size, output_size, dropout_p=0.1):
        super(AttnDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        
        self.hidden_layer = nn.Linear(input_size, input_size) 
        # since we use last hidden state as input and want to apply dot prodcut of result and hidden states of encoder 

        #self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(self.input_size * 2, self.output_size)

    def forward(self, input, encoder_outputs):
        x = F.relu(self.hidden_layer(input))

        attention_scores = []
        for encoder_hidden_state in encoder_outputs:
            attention_scores.append(torch.dot(encoder_hidden_state, x[0][0]))
        
        attention_scores = torch.FloatTensor(attention_scores)
        attention_distribution = F.softmax(attention_scores, dim=0)
        attn_applied = torch.bmm(attention_distribution.unsqueeze(0).unsqueeze(0), encoder_outputs.unsqueeze(0))
        
        concat = torch.cat((x[0], attn_applied[0]), 1)
        output = F.softmax(self.out(concat), dim=1)
        return output


class SequencePadder():
    def __init__(self, symbol) -> None:
        self.symbol = symbol

    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].size(0), reverse=True)
        sequences = [x[0] for x in sorted_batch]
        labels = [x[1] for x in sorted_batch]
        padded = pad_sequence(sequences, padding_value=self.symbol)
        lengths = torch.LongTensor([len(x) for x in sequences])
        return padded, torch.LongTensor(labels), lengths

class PhrasesClassLoader(Dataset):
    def __init__(self, phrases_df, vocab, embeddings, label_col='Sentiment') -> None:
        super().__init__()
        self.df = phrases_df
        #if label_col in ['Sentiment', 'category']:
        #    vc = self.df[label_col].value_counts()
        #    self.labels = self.df[label_col].apply(
        #        lambda x: 0 if x  == vc.index.values[0] else 1).values
        self.labels = self.df[label_col].values

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.word2idx = {w: idx for (idx, w) in enumerate(sorted(self.vocab))}
        self.idx2word = {idx: w for (idx, w) in enumerate(sorted(self.vocab))}
        self.data = []
        for phrase in self.df['Phrase'].values:
            self.data.append(torch.stack([torch.tensor(self.word2idx[w],
                                          dtype=torch.long) for w in phrase.split()]))
            #self.data.append(torch.stack([torch.tensor(w,
            #                              dtype=torch.StringType) for w in phrase.split()]))


    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
         if torch.is_tensor(idx):
             idx = idx.tolist()
         return self.data[idx], self.labels[idx]

# %%
pad_sym = '<pad>'
vocab = np.append(vocab, pad_sym)
loader = PhrasesClassLoader(df, vocab, glove)

# %%
vectors = []
for word in vocab:
    vectors.append(glove[word])

# %%
print(len(vectors[1]))

# %%
criterion = nn.CrossEntropyLoss()
encoder_model = MyGRU(input_size=len(vocab), 
              embedding_size=50,
              hidden_size=2,
              output_size=5, 
              vectors=vectors)

decoder_model = AttnDecoder(input_size=2, output_size=5)

encoder_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder_model.parameters()))
decoder_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, decoder_model.parameters()))

# %%
encoder_model.train()
decoder_model.train()

max_length = 20

for epoch in range(2):
    running_loss = 0
    mini_batch_nr = 0

    for data, labels, lens in DataLoader(loader, batch_size=1,
                    collate_fn=SequencePadder(loader.word2idx['<pad>'])):
        # do stuff here
        # hint: collate_fn is a function that operates on each batch.
        # This one handles padding for us
        encoder_optim.zero_grad()
        decoder_optim.zero_grad()
        encoder_hidden = encoder_model.init_hidden(batch_size=1)
        
        encoder_output = None
        single_data_point = data[0][0]

        encoder_hidden_states = torch.zeros(max_length, 2)

        loss = 0

        for i in range(lens):
            encoder_output, encoder_hidden = encoder_model(data[i], encoder_hidden)
            encoder_hidden_states[i] = encoder_output[0,0]

        output = decoder_model(encoder_hidden, encoder_hidden_states)

        loss = criterion(output, torch.LongTensor(labels))
        loss.backward() # Does backpropagation and calculates gradients
        
        encoder_optim.step() # Updates the weights accordingly
        decoder_optim.step()

        # print statistics
        running_loss += loss.item()
        if mini_batch_nr % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, mini_batch_nr + 1, running_loss / 2000))
            running_loss = 0.0
        
        mini_batch_nr += 1

# %%
