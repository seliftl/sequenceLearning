#%%
from torch import nn
import torch
import pandas as pd
import string
from HanTa import HanoverTagger as ht
import matplotlib.pyplot as plt 
import re
import numpy as np
from torch import optim
from torch.optim import optimizer
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, dataloader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence, PackedSequence

#%%
def load_thesis_data(path='../res/theses.tsv', remove_outlier=True):
    df = pd.read_csv(path, sep='\t',
                     names=['year', 'category', 'type', 'title'])

    df = df[df['type'] != 'Diplom'] 
    df = df[df['category'] != 'im Ausland']
    tagger = ht.HanoverTagger('morphmodel_ger.pgz')
    def process_title(title):
        remove_pun = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        remove_digits = str.maketrans(string.digits, ' '*len(string.digits))
        title = title.translate(remove_digits)
        title = title.translate(remove_pun)
        title = re.sub(' {2,}', ' ', title)
        title = ' '.join([lemma for _, lemma, _ in tagger.tag_sent(title.split(' '), casesensitive=False)])
        return title.lower()

    df['title'] = df['title'].apply(process_title)
    df['length'] = df['title'].apply(lambda x: len(x.split()))
    if remove_outlier:
        df = df[df['length'].between(4, 20)]
    vocab = df['title'].str.split(expand=True).stack().value_counts().index.values

    return df, vocab

class MyGRU(nn.Module):
    def __init__(self,
                input_size,
                embedding_size,
                hidden_size,
                output_size, # number of classes
                num_layers=1,
                bidirectional=False,
                vectors=None, 
                dropout = 0.3):

        super(MyGRU, self).__init__()
        self.input_size = input_size # vocabulary size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        if vectors is not None:
            self.embedding = nn.Embedding(
                            num_embeddings=len(vectors),
                            embedding_dim=embedding_size).from_pretrained(
                            torch.cat(list(vectors.values())).reshape(-1, embedding_size))
        else:
        # embedding layer as input
            self.embedding = nn.Embedding(num_embeddings=input_size,
                                          embedding_dim=embedding_size)
        self.rnn = nn.GRU(
                        input_size=embedding_size,
                        hidden_size=hidden_size,
                        num_layers=self.num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional,
        )

        self.num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(self.hidden_size * self.num_directions, output_size)
        self.word_context_vector = nn.Linear(output_size, 1, bias=False)  
        self.dropout = nn.Dropout(dropout)  	
        
    def forward(self, x, lengths, h_n=None):
        sentences = self.dropout(self.embedding(x)) 
        # Re-arrange as words by removing word-pads (SENTENCES -> WORDS)
        packed_words = pack_padded_sequence(sentences,
                                            lengths=lengths.tolist(),
                                            batch_first=False,
                                            enforce_sorted=False) 
        # Apply the word-level RNN over the word embeddings (PyTorch automatically applies it on the PackedSequence)
        packed_words, h_n = self.rnn(packed_words)  
        # Find attention vectors by applying the attention linear layer on the output of the RNN
        att_w = self.fc(packed_words.data)  
        att_w = torch.tanh(att_w)  
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_w = self.word_context_vector(att_w).squeeze(1) 

        # Compute softmax over the dot-product manually
        # First, take the exponent
        max_value = att_w.max()  # scalar, for numerical stability during exponent calculation
        att_w = torch.exp(att_w - max_value)  

        # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
        att_w, _ = pad_packed_sequence(PackedSequence(data=att_w,
                                                      batch_sizes=packed_words.batch_sizes,
                                                      sorted_indices=packed_words.sorted_indices,
                                                      unsorted_indices=packed_words.unsorted_indices),
                                       batch_first=True)  

        # Calculate softmax values as now words are arranged in their respective sentences
        word_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True) 

        # Similarly re-arrange word-level RNN outputs as sentences by re-padding with 0s (WORDS -> SENTENCES)
        sentences, _ = pad_packed_sequence(packed_words, batch_first=True) 

        # Find sentence embeddings
        sentences = sentences * word_alphas.unsqueeze(2)  
        sentences = sentences.sum(dim=1)  
        return sentences, word_alphas

    def init_hidden(self, batch_size=1):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        h_dim_0 = self.num_layers * self.num_directions
        hidden = torch.zeros(h_dim_0, batch_size, self.hidden_size, device=device)
        return hidden


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


class ThesisClassLoader(Dataset):
    def __init__(self, theses_df, vocab, label_col='type') -> None:
        super().__init__()
        self.df = theses_df        
        if label_col in ['type', 'category']:
            vc = self.df[label_col].value_counts()
            self.labels = self.df[label_col].apply(
                lambda x: 0 if x  == vc.index.values[0] else 1).values
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.word2idx = {w: idx for (idx, w) in enumerate(sorted(self.vocab))}
        self.idx2word = {idx: w for (idx, w) in enumerate(sorted(self.vocab))}
        
        self.data = []
        for title in self.df['title'].values:
            self.data.append(torch.stack([torch.tensor(self.word2idx[w],
                                          dtype=torch.long) for w in title.split()]))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
         if torch.is_tensor(idx):
             idx = idx.tolist()
         return self.data[idx], self.labels[idx]


def train(data_loaders, model, criterion, optimizer, num_epochs):
    for epoch in range(1, num_epochs +1):
        print('Epoch {}/{}'.format(epoch, num_epochs))          

        for phase in ['train', 'test']:
            is_train = phase == 'train'

            # training mode enables dropout
            model.train() if is_train else model.eval()
            running_loss = 0.0
            running_corrects = 0
            important_words = {}
            class_zero = []
            class_one = []
            # Batches
            for inputs, labels, lens in DataLoader(data_loaders[phase], batch_size=32, collate_fn=SequencePadder(data_loaders[phase].word2idx['<pad>'])):
                inputs = inputs.to(device)  
                labels = labels.to(device)  
                # Forward prop.
                scores, word_alphas = model(inputs, lens) 

                # get words with most impact
                if is_train == False:                    
                    for attention in word_alphas:
                        attention_list = attention.detach().numpy()
                        max_value = max(attention_list)
                        max_index = [i for i, j in enumerate(attention_list) if j == max_value]
                    inputs= inputs.transpose(0,1)
                    for i in range(0, len(labels)): 
                        words = []
                        for w in inputs[i]:
                            words.append(data_loaders[phase].idx2word[w.item()])
                        if(labels[i].item() == 0):
                            class_zero.append(words[max_index[0]])
                        else:
                            class_one.append(words[max_index[0]])
                # Loss
                loss = criterion(scores, labels)  

                # Back prop.
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Find accuracy
                _, predictions = scores.max(dim=1)  
                running_corrects += torch.eq(predictions, labels).sum().item()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(data_loaders[phase].data)
            epoch_acc = running_corrects / len(data_loaders[phase].data)
            # Print status
            print('Loss {:.4f} Accuracy {:.3f}'.format(epoch_loss,epoch_acc))
    important_words[0] = class_zero
    important_words[1] = class_one
    print('Most important words for classes:', important_words)
    return epoch_acc

# define parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df, vocab = load_thesis_data()
pad_sym = '<pad>'
vocab = np.append(vocab, pad_sym)
n_epochs = 20
hidden_size = 64
embedding_size = 60
n_layers = 1
bi_direct = False
lr = 0.0001

kf = KFold(n_splits = 5)
fold = 1
results = {}

for train_index, test_index in kf.split(df):
    datasets = ['train', 'test']
    model = MyGRU(len(vocab), embedding_size, hidden_size, num_layers=n_layers, output_size=2, bidirectional=bi_direct)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    data_loaders = {
        'train': ThesisClassLoader(df.iloc[train_index], vocab),
        'test': ThesisClassLoader(df.iloc[test_index], vocab)
    }
    acc = train(data_loaders, model, criterion=criterion, optimizer = optimizer, num_epochs=n_epochs)
    results[fold] = acc
    fold += 1
print(results)
# %%
def visualize_random_title():
    loader = ThesisClassLoader(df.sample(), vocab)
    for inputs, labels, lens in DataLoader(loader, batch_size=1, collate_fn=SequencePadder(loader.word2idx['<pad>'])):
                inputs = inputs.to(device)  
                labels = labels.to(device)  
                
                # Forward prop.
                scores, word_alphas = model(inputs, lens) 
                inputs= inputs.transpose(0,1)
                for input in inputs: 
                    words = []
                    for w in input:
                        words.append(loader.idx2word[w.item()])
            
                for attention in word_alphas:
                    plot_labels = words
                    
                    fig = plt.figure(figsize=(12,4))
                    plt.xticks(range(len(attention)), plot_labels)
                    plt.ylabel('Impact')
                    plt.bar(range(len(attention)), attention.detach().numpy()) 
                    plt.show()  
                
                _, predictions = scores.max(dim=1) 
                print(torch.eq(predictions, labels).item())
visualize_random_title()
# %%
