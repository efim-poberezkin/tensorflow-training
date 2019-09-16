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

# %%
df = pd.read_csv("data/multilabel_dataset.csv")
df.describe(include="all")

# %%
features = df.iloc[:, :-14]
labels = df.iloc[:, -14:]
cv = KFold(n_splits=5, random_state=37)

# %% [markdown]
# # scikit-learn baseline

# %%
lr = LogisticRegression(solver="lbfgs")
clf = OneVsRestClassifier(lr)
scores = []
for train_index, test_index in cv.split(features):
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = f1_score(y_test, y_pred, average="micro")
    scores.append(score)
print(mean(scores))

# %% [markdown]
# # pure tensorflow

# %%
tf.random.set_random_seed(47)

# Model parameters
learning_rate = 0.03
n_epochs = 100

# Dimensions
num_features = len(features.columns)
num_labels = len(labels.columns)

# %%
# Create placeholders for features and labels
X = tf.placeholder(tf.float32, name="features")
Y = tf.placeholder(tf.float32, name="labels")

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
logits = tf.matmul(X, w) + b

# Define loss function. Unlike the single-label case, we should not output
# a softmax probability distribultion as labels are classified independently.
# Instead we apply a sigmoid on the logits as they are independent logistic regressions.
# Since we treat each logit as an independent logistic regression, we need to sum
# so that the whole model's performance is the sum of its per-class performances
loss = tf.reduce_mean(
    tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y), axis=1
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
for train_index, test_index in cv.split(features):
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())

        # Train model
        for epoch in range(n_epochs):
            _, l = sess.run([optimizer, loss], feed_dict={X: X_train, Y: y_train})

        # Calculate predicted values
        y_pred = sess.run(one_hot_prediction, {X: X_test, Y: y_test})

    score = f1_score(y_test, y_pred, average="micro")
    scores.append(score)
    
print(mean(scores))
