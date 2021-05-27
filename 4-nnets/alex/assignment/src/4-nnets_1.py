#%%
import numpy as np
from numpy.linalg import norm
import io
import torch
from torch import nn
from torch.autograd import Variable
import torch.functional as F
import torch.nn.functional as F
from scipy import spatial
import random
from torch.utils.data import DataLoader, TensorDataset

#%%
# -----------------------------------------------------------------
# Task 1: Skip-Gram
# ----Part 2 of assignment: Generate embeddings from thesis.tsv----

# Load the data
def load_data() -> list:
    file = io.open("../res/theses.tsv", mode="r", encoding="utf-8")
    theses = file.readlines()
    theses = ['<s> ' + x.split('\t')[3].strip().lower() for x in theses] 
    return theses

theses = load_data()
print(theses[0])

#reduce theses for testing purposes
#theses = theses[:200]

# %%
# Tokenize thesis titles
def tokenize(corpus):
    tokens = [x.split() for x in corpus]
    return tokens

tokenized_corpus = tokenize(theses)
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
#most_sim_words_cloud = get_most_n_similar_words(n, 'cloud')
#print(most_sim_words_cloud)
#most_sim_words_virtuelle = get_most_n_similar_words(n, 'virtuelle')
#print(most_sim_words_virtuelle)


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
# -----------------------------------------------------------------
# Task 2: RNN Language Model
# Partly used tutorial: https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
# %%
#----------- Part 1: Add padding to sequences -----------
pad_symbol = '#PAD#'
# Define helper methods
def find_longest_sequences(sequences):
    longest_length = 0

    for sequence in sequences:
        cur_length = len(sequence)
        if longest_length < cur_length:
            longest_length = cur_length

    return longest_length

def add_padding_to_sequences(longest_length, sequences):
    for sequence in sequences:
        if len(sequence) < longest_length:
            number_of_appends = longest_length - len(sequence)
            to_extend = [pad_symbol] * number_of_appends
            sequence.extend(to_extend)

# %%
# Add padding to sequences and add pad_symbol to vocabulary
longest_length = find_longest_sequences(tokenized_corpus)
add_padding_to_sequences(longest_length, tokenized_corpus)

padding_symbol_idx = len(vocabulary)
vocabulary.append(pad_symbol)
word2idx[pad_symbol] = padding_symbol_idx
idx2word[padding_symbol_idx] = pad_symbol

print(tokenized_corpus[0])

# %%
# ----------- Part 2: Generate one-hot-encoded-vectors -----------
#
#one_hot_encoded_vecs = []
#
#for i in range(len(vocabulary)):
#    vec = np.zeros(len(vocabulary))
#    vec[i] = 1.0
#    one_hot_encoded_vecs.append(vec)
#
#print(one_hot_encoded_vecs[0])

# ----------- Part 3: Prepare embeddings and targets for training -----------
# %%
embeddings = W2.cpu().detach().numpy()

# Add 0 embedding for padding symbol
embeddings = np.append(embeddings, [np.zeros(embedding_dims)], axis=0)
embedding2idx = {}
for index, embedding in enumerate(embeddings):
    embedding2idx[np.array2string(embedding)] = index

input_sequences = []
target_sequences = []


for title in tokenized_corpus:
    input_sequence = []
    target_sequence = []
    for i, word in enumerate(title):
        idx = word2idx[word]

        # All words besides last need to be added as embeddings to input
        if i != len(title)-1:
            input_sequence.append(embeddings[idx])
        
        # All words besides first need to be added as one-hot-encoded-vecs to target
        if i != 0:
            #target_sequence.append(one_hot_encoded_vecs[idx])
            target_sequence.append(idx)

    input_sequences.append(input_sequence)
    target_sequences.append(target_sequence)

print(tokenized_corpus[0])
print(input_sequences[0])
print(target_sequences[0])


# %%
# Set device for PyTorch
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# %%
# ----------- Part 4: Define the rnn-model -----------
class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):    
        batch_size = x.size(0)

        #Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden

# %%
# --------------- RNN-LM: Exercies 1: Validation of RNN-LM ----------------
def train_model(train_embedding_sequences, train_target_sequences):
    model = Model(input_size=embedding_dims, output_size=len(vocabulary), hidden_dim=32, n_layers=1)
    # Set model to defined device
    model.to(device)

    # Define hyperparameters
    n_epochs = 10
    lr=0.01

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Prepare Training Data
    train_input_sequences = np.array(train_embedding_sequences, dtype=np.float32)
    train_input_sequences = torch.from_numpy(train_input_sequences)

    # Flatten Training Targets
    train_target = []
    for title in train_target_sequences:
        for target_index in title:
            train_target.append(target_index)

    train_target = torch.Tensor(train_target)

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad() # Clears existing gradients from previous epoch

        #train_input_sequences.to(device)
        output, hidden = model(train_input_sequences)
        output = output.to(device)
        train_target = train_target.to(device)

        loss = criterion(output, train_target.view(-1).long())
        loss.backward() # Does backpropagation and calculates gradients
        optimizer.step() # Updates the weights accordingly
        
        if epoch%10 == 0:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))

    return model


def predict(model, words):
    if len(words) == 0:
        words = ['<s>']

    input_embeddings = np.array([[embeddings[word2idx[w]] for w in words]], dtype=np.float32)
    input_embeddings = torch.from_numpy(input_embeddings)
    input_embeddings.to(device)
    
    out, hidden = model(input_embeddings)

    probability_distribution = nn.functional.softmax(out[-1], dim=0).data

    # Taking the class with the highest probability score from the output
    # word_idx = torch.max(prob, dim=0)[1].item()

    return probability_distribution

def calc_perplexity(model, test_data):
    probability_of_model = 1
    word_count = 0
    for title in test_data:
        context = []
        for embedding in title:
            word_index = embedding2idx[np.array2string(embedding)]
            word = idx2word[word_index]
            if word == '<s>':
                continue
            if word == pad_symbol:
                break
            
            word_count += 1

            probability_distribution = predict(model, context)
            probability_of_model *= probability_distribution[word_index]

            context.append(word)
    
    perplexity = (1/probability_of_model)**(1/float(word_count))
    return perplexity

# %%
def split_data_in_k_parts(k, input_data, target_data):
    size_of_part = len(tokenized_corpus)//k
    cur_index = 0
    input_data_parts = []
    target_data_parts = []
    
    for i in range(1,k+1):
        input_data_parts.append(input_data[cur_index:cur_index+size_of_part])
        target_data_parts.append(target_data[cur_index:cur_index+size_of_part])
        cur_index += size_of_part
    
    return input_data_parts, target_data_parts
        
# %%
# 5 fold validation
seperated_input_sequences, seperated_training_sequences = split_data_in_k_parts(5, input_sequences, target_sequences)
summed_perplexity = 0

for k in range(5):
    # Set training and test data
    train_input_seq = []
    train_target_seq = []
    
    test_input_seq = seperated_input_sequences[k]
            
    for i in range(5):
        if i != k:
            if len(train_input_seq) == 0:
                train_input_seq = seperated_input_sequences[i]
                train_target_seq = seperated_training_sequences[i]
            else:
                train_input_seq.extend(seperated_input_sequences[i])
                train_target_seq.extend(seperated_training_sequences[i])

    # Train model
    model = train_model(train_input_seq, train_target_seq)
    # Evaluate model
    perplexity = calc_perplexity(model, test_input_seq)
    
    summed_perplexity += perplexity

print('Overall Perplexity:', summed_perplexity/5)


# %%
# --------------- RNN-LM: Exercies 2: Sampling from titles ----------------
def sample(model, out_len, start=''):
    model.eval() # eval mode
    start = start.lower()
    title = start.split()

    size = out_len - len(title)

    # Now pass in the previous characters and get a new one
    for i in range(size):
        propability_distribution = predict(model, title)
        word_ranking = np.random.choice(list(word2idx.keys()), 1, propability_distribution)
        next_word = word_ranking[0]
        if next_word == pad_symbol:
            next_word = word_ranking[1]

        title.append(next_word)

    return title

# %%
print(sample(model, 5, 'Konzeption'))
