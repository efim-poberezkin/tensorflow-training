# -*- coding: utf-8 -*-
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

# %% [markdown] {"toc-hr-collapsed": false}
# # User rating based on the review - TF-IDF

# %% [markdown]
# ## Imports

# %%
# Standard library
import pickle
import re
from string import punctuation

# Preprocessing
import nltk
nltk.download("stopwords", quiet=True)

import progressbar
import stop_words
from pymystem3 import Mystem

# Math and visualization
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("white")
sns.set_context("notebook", font_scale=1.1)

import vapeplot
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from wordcloud import WordCloud

# Training and evaluation
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Misc
from warnings import filterwarnings
filterwarnings("ignore")

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# %% [markdown]
# ## Preparation

# %%
with open("auxiliary/reviews_dataset.pkl", "rb") as f:
    X, y = pickle.load(f)

X[:6]
y[:6]

# %% [markdown]
# ### Preprocessing

# %%
# Initializing stuff beforehand for performance.
# When stripping punctuation, we'll use a whitespace and not an empty string in the table
# to avoid concatenating words when whitespace is missing after a punctuation mark
table = str.maketrans({key: " " for key in punctuation})

# Create lemmatizer and stopwords list
mystem = Mystem()

# Compile list of stopwords
stops = set(
    nltk.corpus.stopwords.words("russian")
    + stop_words.get_stop_words("russian")
    + ["свой"]
)


def preprocess(text):
    # Strip punctuation
    text = text.translate(table)

    # Convert all numbers to empty strings
    text = re.sub(r"\d+", "", text)

    tokens = mystem.lemmatize(text.lower())
    tokens = [
        token
        for token in tokens
        if token not in stops and token != " " and token.strip() not in punctuation
    ]
    text = " ".join(tokens)
    return text


# %%
X_prep = []

bar = progressbar.ProgressBar(
    maxval=len(X),
    widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.SimpleProgress()],
)
_ = bar.start()
for idx, item in enumerate(X):
    X_prep.append(preprocess(item))
    bar.update(idx + 1)
bar.finish()

X = X_prep

# %%
X[:6]

# %% [markdown]
# Let's dump our preprocessed text data to a separate pickle file since we might have to reuse it later.

# %%
with open("auxiliary/reviews_dataset_preprocessed.pkl", "wb") as f:
    pickle.dump(X, f)

# %% [markdown]
# ### TF-IDF representation

# %%
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X)
features = np.array(tfidf.get_feature_names())


# %% [markdown] {"toc-hr-collapsed": false}
# ## Word importance

# %% [markdown]
# Let's take the look at the most important words across the whole corpus.

# %%
def top_mean_feats_for_class(X_tfidf, y, features, clss, top_n=25):
    """Returns the top n features that on average are most important amongst documents
    for a given class."""
    ids = np.where(np.array(y) == clss)
    feats_df = top_mean_feats(X_tfidf, features, ids, top_n=top_n)
    return feats_df


def top_mean_feats(X_tfidf, features, grp_ids=None, top_n=25):
    """Returns the top n features that on average are most important amongst documents in rows
    indentified by indices in grp_ids."""
    if grp_ids:
        D = X_tfidf[grp_ids]
    else:
        D = X_tfidf

    # Finding averages only of non-zero elements of sparse matrix turned out to make less sense
    # than finding overall means when visualizing results, because in that case features
    # that on average are most important seem to be very specific and rare
    tfidf_means = np.squeeze(np.asarray(np.mean(D, axis=0)))
    return top_tfidf_feats(tfidf_means, features, top_n)


def top_tfidf_feats(row, features, top_n=25):
    """Gets top n tfidf values in row and return them with their corresponding feature names."""
    topn_ids = np.argsort(row.data).flatten()[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ["feature", "tfidf"]
    return df


# %%
N_BARPLOT = 25
N_WORDCLOUD = 200
BARPLOT_DIMS = (11.7, 8.27)
WORDCLOUD_DIMS = (10, 10)


def generate_bar_plot(feats_df, colormap=None):
    fig, ax = plt.subplots(figsize=BARPLOT_DIMS)
    sns.barplot(x="tfidf", y="feature", data=feats_df, orient="h", palette=colormap)
    ax.set_title(f"{N_BARPLOT} Most Important Words")
    ax.set(xlabel="Mean TF-IDF", ylabel="")


def generate_wordcloud(feats_df, mask=None, colormap=None, contour_color=None):
    wc = WordCloud(
        background_color="white",
        width=1024,
        height=720,
        mask=mask,
        max_words=N_WORDCLOUD,
        colormap=colormap,
        contour_width=5,
        contour_color=contour_color,
    )
    wc_array = wc.generate_from_frequencies(
        dict(zip(feats_df.feature, feats_df.tfidf))
    ).to_array()

    fig, ax = plt.subplots(figsize=WORDCLOUD_DIMS)
    plt.imshow(wc_array, interpolation="bilinear")
    ax.axis("off")


# %% [markdown]
# ### Negative reviews

# %%
neg_feats_df = top_mean_feats_for_class(X_tfidf, y, features, clss=1, top_n=N_WORDCLOUD)

# %%
cmap = sns.blend_palette(vapeplot.palette("crystal_pepsi"))
generate_bar_plot(neg_feats_df[:N_BARPLOT], colormap=cmap)

# %%
sad_mask = np.array(Image.open("wordcloud-masks/sad-smiley.png"))
cmap = LinearSegmentedColormap.from_list(_, vapeplot.palette("cool"))
generate_wordcloud(neg_feats_df, mask=sad_mask, colormap=cmap, contour_color="#94D0FF")

# %% [markdown]
# ### Positive reviews

# %%
pos_feats_df = top_mean_feats_for_class(X_tfidf, y, features, clss=5, top_n=N_WORDCLOUD)

# %%
cmap = sns.blend_palette(vapeplot.palette("crystal_pepsi"))
generate_bar_plot(pos_feats_df[:N_BARPLOT], colormap=cmap)

# %%
happy_mask = np.array(Image.open("wordcloud-masks/happy-smiley.png"))
cmap = LinearSegmentedColormap.from_list(_, vapeplot.palette("vaporwave"))
generate_wordcloud(pos_feats_df, mask=happy_mask, colormap=cmap, contour_color="#8795E8")

# %% [markdown]
# ## Training

# %% [markdown]
# Now let's train our model on TF-IDF representation and check its performance on the test set.
#
# We won't be using cross validation here, so that when we'll be training neural nets for the same task later on we'll have the same training and evaluation setup, and cross validating NNs training would take too long.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=37, stratify=y
)

# %%
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)

model = LogisticRegression(solver="lbfgs", multi_class="multinomial")
model.fit(X_train_tfidf, y_train)

# %%
text_clf = Pipeline([("tfidf", tfidf), ("model", model)])
y_pred = text_clf.predict(X_test)
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")

# %%
print(metrics.classification_report(y_test, y_pred))

# %% [markdown]
# We can see that our model is more accurate on very negative and especially very positive reviews and less so on the ones in the middle, which is explainable since reviews with average scores are probably expected to have borders blurred between them in terms of sentimental coloration, which makes it harder for a model to make a correct prediction.
