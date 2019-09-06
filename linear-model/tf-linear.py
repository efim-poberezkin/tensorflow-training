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
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier

# %%
df = pd.read_csv("../auxiliary/multilabel_dataset.csv")
df.describe(include="all")

# %%
features = df.iloc[:, :-14]
targets = df.iloc[:, -14:]

# %% [markdown]
# # scikit-learn baseline

# %%
lr = LogisticRegression(solver="lbfgs")
clf = OneVsRestClassifier(lr)
scores = cross_val_score(clf, features, targets, scoring="f1_micro", cv=5)
print(scores.mean())
