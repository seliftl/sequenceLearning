# Assignment 3: Hidden Markov Models

In this assignment, we'll be revising word recognition, this time using hidden Markov models.
As with [assignment 1](https://github.com/seqlrn/1-dynamic-programming) (part 4), we'll be using the [free spoken digits](https://github.com/Jakobovski/free-spoken-digit-dataset) dataset.
We will be using the [`hmmlearn`](https://hmmlearn.readthedocs.io/en/latest/index.html) library (which depends on numpy) 

Please use [hmms.py](src/1-basics.py) for the whole assignment.
When submitting your work, please do **not** include the dataset.


## Basic Setup

As you can learn from the [tutorial](https://hmmlearn.readthedocs.io/en/latest/tutorial.html#), `hmmlearn` provides us with the base implementation of hidden Markov models; we'll be using the `hmm.GaussianHMM`, which implements HMMs with a single Gaussian emission probability per state.
For a starter, build a basic isolated word recognizer that uses a separate model for each digit.

1. Compute the MFCC features for the complete data set (3000 recordings; use `n_mfcc=13`).
2. Implement a 6-fold cross-validation (x/v) loop to (later) figure out, which test speaker performs best/worst.
3. Inside the c/v loop, train an individual HMM with linear topology for each digit.
    - The `fit` expects features to be [sequential in a single array](https://github.com/hmmlearn/hmmlearn/blob/38b3cece4a6297e978a204099ae6a0a99555ec01/lib/hmmlearn/base.py#L439); see `np.concatenate(..., axis=0)`
    - How many states (`n_components`) do you choose, and why?
    - How can you enforce a linear topology?
    - You might find that certain digits perform particularly bad; what could be a reason and how to mitigate it?
4. Compute a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) for each speaker and for the overall dataset.


## Decoding

The example above is can't handle sequences of spoken digits.
In this part of the assignment, you'll build a basic decoder that is able to decode arbitrary sequences of digits (without a prior, though).
The `decode` method in `hmmlearn` only works for a single HMM.
There are two ways how to solve this assignment:

1. Construct a "meta" HMM from the previously trained digit HMMs, by allowing state transitions from one digit to another; the resulting HMM can be decoded using the existing `decode` method (don't forget to re-map the state ids to the originating digit).
2. (Optional) Implement a real (time-synchronous) decoder using beam search. The straight-forward way is to maintain a (sorted) list of active hypotheses (ie. state history and current log-likelihood) that is first expanded and then pruned in each time step. The tricky part is at the "end" of a model: do you loop or expand new words?

Now for this assignment:

1. Generate a few test sequences of random length in in between 3 and 6 digits; use [`numpy.random.randint`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html) and be sure to also retain the digits sequence since we need to compute edit distance between reference and hypotheses later.
2. Combine th epreviously trained HMMs to a single "meta" HMM, altering the transition probabilities to make a circular graph that allows each word to follow another.
3. Implement a method that converts a state sequence relating to the meta HMM into a sequence of actual digits.
3. Decode your test sequences and compute the [word error rate](https://pypi.org/project/jiwer/) (WER)
5. Compute an overall (cross-validated) WER.
6. (Optional) Implement a basic time-synchronous beam search; how do the results compare to the above viterbi decoding in terms of accuracy and time?
