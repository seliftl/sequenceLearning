#%%
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
#%%
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
#%%
# Understanding padding, pad_packed_sequence in combination with data loaders:
df = pd.read_csv('res/theses.tsv',sep='\t',
                 names=['year', 'category', 'type', 'title'])
df['title'] = df.title.str.lower()
vocab = df.title.str.split(expand=True).stack().value_counts().index.values
pad_sym = '<pad>'
vocab = np.append(vocab, pad_sym)
dataset = ThesisClassLoader(df, vocab)

loader = DataLoader(dataset, batch_size=2, collate_fn=SequencePadder(dataset.word2idx[pad_sym]))
embedding_size = 3
inp_size = len(vocab)
hidden_size = 3
output_size = 2
criterion = nn.CrossEntropyLoss()

# num_layers x batch_size x hidden_size
# https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
h_0 = torch.zeros(1, 2, hidden_size)
embedding = nn.Embedding(num_embeddings=inp_size, embedding_dim=embedding_size)
rnn = nn.RNN(embedding_size, hidden_size, num_layers=1)
h2c = nn.Linear(hidden_size, output_size)

for data, lengths, labels in loader:
    # without pack padded sequence:
    x = embedding(data)
    out_unp, h_unp = rnn(x, h_0)
    out_cls = h2c(h_unp)
    loss_unp = criterion(out_cls.view(*out_cls.size()[1:]),
                         torch.LongTensor(labels))
    # with pack padded sequence:
    padded_seq = pack_padded_sequence(x, lengths)
    out_packed, h_packed = rnn(padded_seq, h_0)
    out_packed_unpacked, lens_packed_unpacked = pad_packed_sequence(out_packed)
    out_cls_packed = h2c(h_packed)
    loss_packed = criterion(out_cls_packed.view(*out_cls_packed.size()[1:]),
                            torch.LongTensor(labels))
    # What you would need to change, to get the same result as with packed sequence:
    out_unp_2, h_unp_2 = rnn(x, h_0)
    # instead of: out_cls = h2c(h_unp), indices in hidden states at lengths -1,
    # out_unp_2: has 2 hidden states(batch size 2, meaning 1 hidden state for each sequence),
    # at each time step, index [0, 1] meaning, get hidden state for seq 1 and 2, : meaning all feature
    # dimensions, out_unp_2[lengths -1 [0, 1], :] is the same tensor as h_packed
    out_cls_pick = h2c(out_unp_2[lengths - 1, [0, 1], :])

    loss_pick_h = criterion(out_cls_pick,
                            torch.LongTensor(labels))
    # It does make a difference! Also needs a lot less computation
    # to use packed_sequences
    # Check Matrices, they are the same until the padding has begun
    #lengths[1], because orderd by length and batchsize 2 puts the shorter element at index [1]
    print(out_packed_unpacked[0: lengths[1] + 1])
    print(out_unp[0: lengths[1] + 1])
    # Check Losses
    print('Losses')
    print(loss_pick_h)
    print(loss_packed)
    print(loss_unp)
    break