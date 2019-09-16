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

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier

# %%
df = pd.read_csv("data/multilabel_dataset.csv")
df.describe(include="all")

# %%
X = df.iloc[:, :-14]
y = df.iloc[:, -14:]
cv = KFold(n_splits=5, random_state=37)

# %% [markdown]
# # scikit-learn baseline

# %%
lr = LogisticRegression(solver="lbfgs")
clf = OneVsRestClassifier(lr)
scores = []
for train_index, test_index in cv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = f1_score(y_test, y_pred, average="micro")
    scores.append(score)
print(mean(scores))
