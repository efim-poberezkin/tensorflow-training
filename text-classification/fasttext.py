# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: tensorflow-training
#     language: python
#     name: tensorflow-training
# ---

# %% [markdown]
# # User rating based on the review - fasttext

# %%
import pickle

import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
tf.random.set_random_seed(47)

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

# %%
max_features = 20000
maxlen = 100
batch_size = 128
epochs = 4

# %% [markdown]
# Again we'll load our preprocessed text data.

# %%
with open("auxiliary/reviews_dataset_preprocessed.pkl", "rb") as f:
    X = pickle.load(f)

with open("auxiliary/reviews_dataset.pkl", "rb") as f:
    _, y = pickle.load(f)

X[:5], y[:5]
