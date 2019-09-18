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

# %%
from statistics import mean

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
tf.random.set_random_seed(47)

from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout

# %%
df = pd.read_csv("data/multilabel_dataset.csv")
df.describe(include="all")

# %%
X = df.iloc[:, :-14].values
Y = df.iloc[:, -14:].values
cv = KFold(n_splits=5, random_state=37)

# %% [markdown]
# # scikit-learn baseline

# %%
lr = LogisticRegression(solver="lbfgs")
clf = OneVsRestClassifier(lr)

# %%
scores = []

for train, test in cv.split(X, Y):
    clf.fit(X[train], Y[train])
    Y_pred = clf.predict(X[test])
    score = f1_score(Y[test], Y_pred, average="micro")
    scores.append(score)

print(f"Micro-averaged f1 on cross validation: {mean(scores)}")

# %% [markdown]
# # pure tensorflow

# %%
# Model parameters
learning_rate = 0.03
num_epochs = 100

# Dimensions
num_features = len(X[0])
num_labels = len(Y[0])

# %%
# Create placeholders for features and labels
X_tensor = tf.placeholder(tf.float32, name="features")
Y_tensor = tf.placeholder(tf.float32, name="labels")

# Create variables for weights and bias
w = tf.get_variable(
    shape=(num_features, num_labels),
    initializer=tf.random_normal_initializer(),
    name="weights",
)
b = tf.get_variable(
    shape=(1, num_labels), initializer=tf.zeros_initializer(), name="bias"
)

# Build a model returning logits
logits = tf.matmul(X_tensor, w) + b

# Define loss function. Unlike the single-label case, we should not output
# a softmax probability distribultion as labels are classified independently.
# Instead we apply a sigmoid on the logits as they are independent logistic regressions.
# Since we treat each logit as an independent logistic regression, we need to sum
# so that the whole model's performance is the sum of its per-class performances
loss = tf.reduce_mean(
    tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y_tensor), axis=1
    )
)

# Define training operation
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Make prediction
def multi_label_hot(prediction, threshold=0.5):
    prediction = tf.cast(prediction, tf.float32)
    return tf.cast(tf.greater(prediction, threshold), tf.int64)


prediction = tf.sigmoid(logits)
one_hot_prediction = multi_label_hot(prediction)

# %%
scores = []

for train, test in cv.split(X, Y):
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())

        # Train model
        for epoch in range(num_epochs):
            _, l = sess.run(
                [optimizer, loss], feed_dict={X_tensor: X[train], Y_tensor: Y[train]}
            )

        # Calculate predicted values
        Y_pred = sess.run(one_hot_prediction, {X_tensor: X[test], Y_tensor: Y[test]})

    score = f1_score(Y[test], Y_pred, average="micro")
    scores.append(score)

print(f"Micro-averaged f1 on cross validation: {mean(scores)}")

# %% [markdown]
# # keras

# %%
scores = []

for train, test in cv.split(X, Y):
    # Create and compile model
    model = Sequential()
    model.add(Dense(num_labels, activation="sigmoid", input_shape=(num_features,)))

    adam = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])

    # Fit and make prediction
    model.fit(X[train], Y[train], epochs=num_epochs, batch_size=200, verbose=0)
    Y_pred = (model.predict(X[test]) > 0.5).astype(np.uint8)

    score = f1_score(Y[test], Y_pred, average="micro")
    scores.append(score)

print(f"Micro-averaged f1 on cross validation: {mean(scores)}")

# %% [markdown]
# # keras with nonlinearity

# %%
scores = []

for train, test in cv.split(X, Y):
    # Create and compile model
    model = Sequential()
    model.add(Dense(200, activation="relu", input_shape=(num_features,)))
    model.add(Dropout(0.3))
    model.add(Dense(200, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(200, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(num_labels, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Fit and make prediction
    es = EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=1)
    model.fit(
        X[train],
        Y[train],
        epochs=num_epochs,
        batch_size=200,
        verbose=1,
        validation_split=0.3,
        callbacks=[es]
    )
    Y_pred = (model.predict(X[test]) > 0.5).astype(np.uint8)

    score = f1_score(Y[test], Y_pred, average="micro")
    scores.append(score)

print(f"Micro-averaged f1 on cross validation: {mean(scores)}")
