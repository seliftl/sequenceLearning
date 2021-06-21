#%%
import numpy as np
import pandas as pd

import string
import re

import torch
import random

from HanTa import HanoverTagger as ht

# huggingface
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#%%
# data logistics: load theses title and abstract
# limit_title_len=[4,10] restricts to titles in between 4 and 10 tokens
def load_thesis_data(path='res/theses.tsv', limit_title_len=None):
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    # TODO limit to 'Sprache'==DE and title_len if necessary
    df = pd.read_csv(path, sep='\t',
                     names=['Anmeldedatum', 'JahrAkademisch', 'Art', 'Grad', 'Sprache', 'Titel', 'Abstract'], skiprows=1)
    
    #tagger = ht.HanoverTagger('morphmodel_ger.pgz')
    #def process_title(title):
    #    remove_pun = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    #    remove_digits = str.maketrans(string.digits, ' '*len(string.digits))
    #    title = title.translate(remove_digits)
    #    title = title.translate(remove_pun)
    #    title = re.sub(' {2,}', ' ', title)
    #    title = ' '.join([lemma for _, lemma, _ in tagger.tag_sent(title.split(' '), casesensitive=False)])
    #    return title.lower()
    
    #def process_abstract(abstract):
    #    remove_pun = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    #    remove_digits = str.maketrans(string.digits, ' '*len(string.digits))
    #    abstract = abstract.translate(remove_digits)
    #    abstract = abstract.translate(remove_pun)
    #    abstract = re.sub(' {2,}', ' ', abstract)
    #    abstract = ' '.join([lemma for _, lemma, _ in tagger.tag_sent(abstract.split(' '), casesensitive=False)])
    #    return abstract.lower()

    # df['Verarbeiteter_Titel'] = df['Titel'].apply(process_title)
    # df['Verarbeiteter_Abstract'] = df['Abstract'].apply(process_abstract)

    df = df[df['Sprache'] == 'DE']

    df['Laenge'] = df['Titel'].apply(lambda x: len(x.split()))
    if limit_title_len:
        df = df[df['Laenge'].between(limit_title_len[0], limit_title_len[1])]

    #vocab = df['Phrase'].str.split(expand=True).stack().value_counts().index.values
    #return df, vocab

    return df

#%%
# set up the models; they will download on first time use but this will take some time (1.2 GB)
# https://huggingface.co/transformers/model_doc/auto.html?highlight=autotokenizer#transformers.AutoTokenizer.from_pretrained
# https://huggingface.co/transformers/model_doc/auto.html?highlight=autotokenizer#transformers.AutoModelForSeq2SeqLM.from_pretrained

# TODO tokenizer = ...
# TODO model = ...
tokenizer = AutoTokenizer.from_pretrained("ml6team/mt5-small-german-finetune-mlsum")
model = AutoModelForSeq2SeqLM.from_pretrained("ml6team/mt5-small-german-finetune-mlsum")

#%%
# method for summary generation, using the global model and tokenizer
def generate_summary(model, abstract, num_beams, repetition_penalty,
                    length_penalty, early_stopping, max_output_length):
    # TODO source_encoding = tokenizer(...)
    source_encoding = tokenizer(abstract,
                                max_length=784,
                                padding="max_length",
                                truncation=True,
                                return_attention_mask=True,
                                add_special_tokens=True,
                                return_tensors="pt")

    # TODO generated_ids = model.generate(...)
    generated_ids = model.generate(input_ids=source_encoding["input_ids"],
                                    attention_mask=source_encoding["attention_mask"],
                                    num_beams=num_beams,
                                    max_length=max_output_length,
                                    repetition_penalty=repetition_penalty,
                                    length_penalty=length_penalty,
                                    early_stopping=early_stopping,
                                    use_cache=True
  )

    # TODO ...map to string using tokenizer.decode and return
    preds=[tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) 
            for gen_id in generated_ids]

    return "".join(preds)

# %%
# main program
df = load_thesis_data()
df.head()

#%%
# now use the pre-trained model to generate some short summaries from the 
# abstracts, and compare them to the reference titles

# adjust these values as desired
num_beams = 2
repetition_penalty = 1.0
length_penalty = 2.0
max_output_length = 120

early_stopping = True

# sample from dataset, using abstracts as input to generate short summary (~title)
from IPython.display import HTML, display
def displaysum(summarize, generated, reference):
    display(HTML(f"""<table>
    <tr><td>summarize:</td><td>{summarize}</td></tr>
    <tr><td>generated:</td><td>{generated}</td></tr>
    <tr><td>reference:</td><td>{reference}</td></tr>
    </table>
    """))

for i in [random.randint(0, len(df) - 1) for _ in range(10)]:
    # load the values
    summarize = df.iloc[i].Abstract
    reference = df.iloc[i].Titel

    # TODO generated = generate_summary(...)
    generated = generate_summary(model, summarize, num_beams, repetition_penalty,
                    length_penalty, early_stopping, max_output_length)

    displaysum(summarize, generated, reference)

# %%
# Task 2: Fine-tuning

# As you could see, the summary quality is pretty much hit-or-miss. Let's use
# a good share of the data to fine-tune the pre-trained model to our task.

from torch.utils.data import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

# %%
class ThesisDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_len, max_output_len):
        self.tokenizer = tokenizer
        self.source_len = max_input_len
        self.summ_len = max_output_len
        self.Titel = df.Titel

        # T5 requires us to prepend the task
        self.Abstract = 'summarize: ' + df.Abstract

    def __len__(self):
        return len(self.Titel)

    def __getitem__(self, index):
        abstract = str(self.Abstract[index])
        title = str(self.Titel[index])

        # TODO use tokenizer.batch_encode_plus to also get the masking
        source_tok = self.tokenizer.batch_encode_plus(...)
        label_tok = self.tokenizer.batch_encode_plus(...)

        input_ids = source_tok['input_ids'].squeeze()
        input_mask = source_tok['attention_mask'].squeeze()
        label_ids = label_tok['input_ids'].squeeze()
        label_mask = label_tok['attention_mask'].squeeze()

        return {
            'input_ids': input_ids.to(dtype=torch.long), 
            'input_mask': input_mask.to(dtype=torch.long), 
            'label_ids': label_ids.to(dtype=torch.long),
            'label_mask': label_mask.to(dtype=torch.long)
        }

# %%

# for each point in the data loader, compute the forward pass, loss and
# backward pass

def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()

    for i, data in enumerate(loader, 0):
        y = data['label_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()

        # set the padding symbols to -100 to be ignored by torch
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

        inputs = data['input_ids'].to(device, dtype=torch.long)
        mask = data['input_mask'].to(device, dtype=torch.long)

        # TODO compute forward pass
        outputs = model(...)

        loss = outputs[0]

        if i % 10 == 0:
            print({"Training Loss": loss.item()})

        # TODO reset optimizer, do backwards pass and optimizer step    
    
    print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        

# %%

# for validation, set the model to eval mode and compute all predictions
def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            y = data['label_ids'].to(device, dtype=torch.long)
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['input_mask'].to(device, dtype=torch.long)

            # TODO make prediction
            generated_ids = model.generate(...)
            
            # TODO use tokenizer.decode to get predicted and target string
            preds = ...
            target = ...
            
            if i % 100 == 0:
                print(f'Completed {i}')

            predictions.extend(preds)
            actuals.extend(target)
    
    return predictions, actuals

#%%

# defining some parameters that will be used later on in the training  
batch_size_train = 32
batch_size_vali = 4

max_input_len = 512    # 512?
max_output_len = 120

# set random seeds and deterministic pytorch for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

# TODO verify: tokenizer and df still loaded and current?

# split the dataframe into training and validation
df_train = df.sample(frac=0.8, random_state=seed)
df_vali = df.drop(df_train.index).reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
print("df={df.shape}, train={df_train.shape}, vali={df_vali.shape}")

# Creating the Training and Validation dataset for further creation of Dataloader
ds_train = ThesisDataset(df_train, tokenizer, max_input_len, max_output_len)
ds_vali = ThesisDataset(df_vali, tokenizer, max_input_len, max_output_len)

# create data loaders for training and validation
from torch.utils.data import DataLoader
dl_train = DataLoader(ds_train, shuffle=True, num_workers=0, batch_size=batch_size_train)
dl_vali = DataLoader(ds_vali, shuffle=True, num_workers=0, batch_size=batch_size_vali)

#%%

# we'll start from the same ml6team/mt5-small-german-finetune-mlsum that we
# used before in our baseline experiment; we will reload it below so that we
# maintain the base model
base = model

# this time, we'll load it explicitly as a T5ForConditionalGeneration; the
# tokenizer will be the same

from transformers import T5ForConditionalGeneration
model = T5ForConditionalGeneration.from_pretrained("ml6team/mt5-small-german-finetune-mlsum")
model = model.to(device)

# Defining the optimizer that will be used to tune the weights of the network in the training session. 
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

#%% Run training loop
epochs_train = 3
epochs_vali = [1, 2, 3]

models = []

for epoch in range(epochs_train):
    # TODO call the training routine from above
    train(...)

    # save the model after each epoch; warning: model size is ~1.2G
    #model.save_pretrained('res/mt5-small-fine-tune-'+epoch)

    if epoch in epochs_vali:
        # TODO call the vali routine from above to generate some summaries
        predictions, actuals = validate(...)

        # display some...
        for i in [random.randint(0, len(predictions) - 1) for _ in range(10)]:
        displaysum(None, generated, reference)

#%%

# save last iteration
model.save_pretrained('res/mt5-small-fine-tune-theses')

#%%
# load model and compare outputs
base = AutoModelForSeq2SeqLM.from_pretrained("ml6team/mt5-small-german-finetune-mlsum")
fine = model  # or any other checkpoint from res/mt5-small-fine-tune-...

#%%
# pick some random theses (from df_vali!) and compare the two models
thesis_picks = [random.randint(0, len(df_vali) - 1) for _ in range(10)]
for num, i in enumerate(thesis_picks):
    print()

    # TODO generate a summary with each of the models
    s1 = generate_summary(...)
    s2 = generate_summary(...)
    
    display(HTML(f"""<table>
    <tr><td>summarize:</td><td>{df_vali.iloc[i].Abstract}</td></tr>
    <tr><td>base:</td><td>{s1}</td></tr>
    <tr><td>fine:</td><td>{s2}</td></tr>
    <tr><td>reference:</td><td>{df_vali.iloc[i].Titel}</td></tr>
    </table>
    """))
