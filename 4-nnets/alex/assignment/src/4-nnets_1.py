#%%
import numpy as np
from numpy.linalg import norm
import io
import torch
from torch.autograd import Variable
import torch.functional as F
import torch.nn.functional as F
from scipy import spatial
import random

#%%
# ----Part 2 of assignment: Generate embeddings from thesis.tsv----

# Load the data
def load_data() -> list:
    file = io.open("../res/theses.tsv", mode="r", encoding="utf-8")
    theses = file.readlines()
    theses = [x.split('\t')[3].strip().lower() for x in theses] 
    return theses

theses = load_data()
print(theses[0])

# %%
# Tokenize thesis titles
def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens

tokenized_corpus = tokenize_corpus(theses)
print(tokenized_corpus[0])

# %%
# Build up the vocabulary
vocabulary = []
for sentence in tokenized_corpus:
    for token in sentence:
        if token not in vocabulary:
            vocabulary.append(token)

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

vocabulary_size = len(vocabulary)

print('Vocab size: ', vocabulary_size)

# %%
# Build up connections between context and center words
window_size = 2
idx_pairs = []
# for each sentence
for sentence in tokenized_corpus:
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
print(idx_pairs[0])

# %%
# Define and run network to generate word embeddings (=> within the matrices)
def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x

embedding_dims = 5
W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
num_epochs = 21
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

# %%
# Check how a word vectors looks like
print(W2[0].detach().cpu().numpy())

# %%
# ----Part 3 of assignment: Analyse embeddings ----

# Definition of helper methods to get similar words
def get_vec_of_word(word):
    word_idx = word2idx[word]
    return W2[word_idx].detach().cpu().numpy()

    
def ret_2nd_ele(tuple_1):
    return tuple_1[1]

def get_most_n_similar_words(n, word):
    word_vec_target_word = get_vec_of_word(word)
    n_most_similar_words = []

    for word_idx, word  in enumerate(vocabulary):
#        word_vec = W2[word_idx].detach().cpu().numpy()
        word_vec = get_vec_of_word(word)
        distance = spatial.distance.cosine(word_vec_target_word, word_vec)

        if len(n_most_similar_words) < n:
            n_most_similar_words.append((word, distance))
        else:
            min_dist_entry = max(n_most_similar_words, key=ret_2nd_ele)
            if min_dist_entry[1] > distance:
                n_most_similar_words[n_most_similar_words.index(min_dist_entry)] = (word, distance)
    
    n_most_similar_words.sort(key=lambda tup: tup[1])
    return n_most_similar_words


# %%
# Get most similar words to 'konzeption', 'cloud' und 'virtuelle'
n = 5
most_sim_words_konzeption = get_most_n_similar_words(n, 'konzeption')
print(most_sim_words_konzeption)
most_sim_words_cloud = get_most_n_similar_words(n, 'cloud')
print(most_sim_words_cloud)
most_sim_words_virtuelle = get_most_n_similar_words(n, 'virtuelle')
print(most_sim_words_virtuelle)


# %%
# ----Part 4 of assignment: Play with embeddings ----

# Helper methods to do dynamic time warping on thesis titles
def get_list_of_word_vecs_for_thesis(thesis_as_tokens):
    word_vecs = []
    for word in thesis_as_tokens:
        word_vecs.append(get_vec_of_word(word))
    return word_vecs

def dtw(obs1: list, obs2: list, sim) -> float:
    cost_matrix = np.full((len(obs1) + 1, len(obs2) + 1), np.inf, dtype=float)
    cost_matrix[0][0] = 0

    for row in range(1, len(obs1) + 1):
        for column in range(1, len(obs2) + 1):
            cost = sim(obs1[row - 1], obs2[column - 1])
            cost_matrix[row, column] = cost + min(cost_matrix[row-1, column],
                                                cost_matrix[row, column-1],
                                                cost_matrix[row-1][column-1])

    return cost_matrix[len(obs1), len(obs2)]

def calc_vec_distance(vec1, vec2):
    return spatial.distance.cosine(vec1, vec2)


# Use dtw to find most similar titles
def get_n_most_similar_thesis(n, thesis_as_tokens):
    word_vecs_target = get_list_of_word_vecs_for_thesis(thesis_as_tokens)
    n_most_similar_thesis = []

    for tokenized_thesis in tokenized_corpus:
        word_vecs = get_list_of_word_vecs_for_thesis(tokenized_thesis)
        distance = dtw(word_vecs_target, word_vecs, calc_vec_distance)

        if len(n_most_similar_thesis) < n:
            n_most_similar_thesis.append((tokenized_thesis, distance))
        else:
            min_dist_entry = max(n_most_similar_thesis, key=ret_2nd_ele)
            if min_dist_entry[1] > distance:
                n_most_similar_thesis[n_most_similar_thesis.index(min_dist_entry)] = (tokenized_thesis, distance)
    
    n_most_similar_thesis.sort(key=lambda tup: tup[1])
    return n_most_similar_thesis

# %%
# Test for some of the titles
for i in range(5):
    index = random.randint(0, len(tokenized_corpus))
    print('Betrachteter Titel: ', tokenized_corpus[index])
    print('Ähnliche Titel: ')
    print(get_n_most_similar_thesis(4, tokenized_corpus[index]))
    print('------------------------------------------------------')
# %%
