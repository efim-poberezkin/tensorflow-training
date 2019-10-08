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
# # User rating based on the review - RNN

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
# cut texts after this number of words (among top max_features most common words)
maxlen = 100
batch_size = 256
epochs = 4

# %% [markdown]
# We've already preprocessed our text data when we were training TF-IDF based classifier, so now we're able to load it and skip preprocessing step.

# %%
with open("auxiliary/reviews_dataset_preprocessed.pkl", "rb") as f:
    X = pickle.load(f)

with open("auxiliary/reviews_dataset.pkl", "rb") as f:
    _, y = pickle.load(f)

X[:5], y[:5]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=37, stratify=y
)

# save an example sentence for later
example_idx = 8
example_review = X_train[example_idx]

# %%
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
print(f"Found {len(word_index)} unique tokens")

# %%
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# %%
# subtract 1 because keras to_categorical() expects integers from 0 to num_classes
y_train = to_categorical(np.array(y_train) - 1)
y_test = to_categorical(np.array(y_test) - 1)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# %% [markdown]
# Here's the final representation of data in a way it'll be fed to our network:

# %%
X_train[example_idx]

# %% [markdown]
# Zeros up to the last 4 numbers is padding. The meaningful indices in the end correspond to words in the following sentence:

# %%
example_review

# %% [markdown]
# Let's make sure it corresponds to the representation above by looking up word indices.

# %%
for word in example_review.split():
    print(f"{word} : {word_index[word]}")

# %%
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
# model.add(Bidirectional(LSTM(64)))
# model.add(Dropout(0.5))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(5, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

# %%
checkpoint = ModelCheckpoint(
    "auxiliary/lstm_best_weights.h5",
    monitor="val_acc",
    verbose=1,
    save_best_only=True,
    mode="max",
)

history = model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=[X_test, y_test],
)
