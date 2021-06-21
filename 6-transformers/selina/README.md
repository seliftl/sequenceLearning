# Transformers

Transformers are powerful attention-based models that have great power for generalization.
Googles T5 is a popular and very recent Text-to-Text transformer.
A cool blog post about it can be found on Googles [AI.blog](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html).
T5 is supposedly able to handle all kinds of NLP tasks including summarization, question answering, and text classification.
The paper sheds some light on [what they actually wanted to achieve](https://arxiv.org/abs/1910.10683).

In this exercise, we want to address the question if we can use a pre-trained transformer model to summarize abstracts of theses so they come close to their original titles?

We will try this task in two variants, but to get started, you'll need to download a pre-trained German T5 model that's already fine-tuned for summarization from [huggingface](https://huggingface.co/ml6team/mt5-small-german-finetune-mlsum).
Note: You can either by clone the model repository or (recommended) use the `transformers` library from huggingface that manages the download (and caching).

```bash
git lfs install  # large file support
git clone https://huggingface.co/ml6team/mt5-small-german-finetune-mlsum
```

```python
# this is the recommended way...
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("ml6team/mt5-small-german-finetune-mlsum")
model = AutoModelForSeq2SeqLM.from_pretrained("ml6team/mt5-small-german-finetune-mlsum")
```

Note that this model will require the `sentencepiece` package for tokenization.


## Task 1: T5 Out of the box

Let's see if we can put the to use in our thesis example from the previous exercises.
Check the instructions for a (similar) [Turkish summarization model](https://huggingface.co/ozcangundes/mt5-small-turkish-summarization) and transfer them to the German model.
Download the `theses.tsv` file containing pairs of German abstracts and titles.

Implement the prediction (ie. the actual summarization) and go through the generated summaries.
Do they convey the main point? Would they make good titles?

Analyze how the parameters `num_beams` (beam size), `repetition_penalty` and `length_penalty` affect the generated outputs.


## Task 2: Fine tune it

The big strength of such large transformer model is that they can be fine tuned to specific tasks or domains.
So lets try that for our abstract-to-thesis-title summarization task using transfer learning, ie. we'll start from the existing model parameters and do a few training iterations using in-domain data.

1. Implement transfer learning of the model you used in the previous task and fine tune the model to our task and data.
2. Re-run the summarization, at this time comparing the summaries of the base model and fine tuned model.

Check the [documentation](https://huggingface.co/transformers/model_doc/t5.html) for what you need to do. 
This [notebook](https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_summarization_wandb.ipynb) might also come in handy.


## Hints

* You'll need the `transformers` and `sentencepiece` packages.
* Remember to clean the data; the provided file contains German titles+abstracts only.
* Tokenization is done/provided by the pre-trained models!
* You may want to remove theses with very short or long titles
