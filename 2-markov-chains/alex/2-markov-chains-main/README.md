# Assignment 2: Markov Chains

In this assignment, we'll be looking at language models as a great example of Markov chains.
Please make sure to install both [`numpy`](https://numpy.org) and [`nltk`](http://www.nltk.org), since we'll be using those for the implementations.


## Trump vs. Obama

Surely, you're aware that the 45th President of the United States (@POTUS45) was an active user of Twitter, until (permanently) banned on Jan 8, 2021.
You can still enjoy his greatness at the [Trump Twitter Archive](https://www.thetrumparchive.com/), where you can download a JSON file linked under FAQs (or see Moodle materials).
We will be using original tweets only, so make sure to remove all retweets.

Another fan of Twitter was Barack Obama (@POTUS43 and @POTUS44), who used the platform in a rather professional way.
There are multiple ways to get the data, but the easiest is to download the [files from Kaggle](https://www.kaggle.com/jayrav13/obama-white-house) (or see Moodle materials).

Please also download the current POTUS (Joe Biden) tweets [from Kaggle](https://www.kaggle.com/rohanrao/joe-biden-tweets) (or see Moodle materials); we will be using those for testing.

Before you get started, please download the files; you can put them into the `res` folder.

In this assignment, you will be doing some Twitter-related preprocessing and training n-gram models to be able to distinguish between Tweets of Trump and Obama.
We will be using NLTK, more specifically it's [`lm`](https://www.nltk.org/api/nltk.lm.html) module.
For details, please go to [potus.py](src/potus.py).


## Theses Inspiration

Imagine you'd have to write another thesis, and you just can't find a good topic to work on.
Well, n-grams to the rescue!
Download the `theses.txt` data set from the Moodle page which consists of ca. 1,000 theses topics chosen by students in the past.

Pay extra attention to preprocessing: how would you handle hyphenated words and acronyms/abbreviations?

In this assignment, you will be sampling from n-grams to generate new potential thesis topics.
For more details, please go to [theses.py](src/theses.py).
