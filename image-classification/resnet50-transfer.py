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
# # Hotdog / Not Hotdog Classification - transfer learning w/ ResNet50

# %%
import numpy as np

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

from keras.applications import resnet
from keras.applications.resnet import ResNet50
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
)
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

# %%
TRAIN_DATA_DIR = "data/train"
VALIDATION_DATA_DIR = "data/test"

BOTTLENECK_TRAIN_PATH = "auxiliary/resnet_bottleneck_train.npy"
BOTTLENECK_VALIDATION_PATH = "auxiliary/resnet_bottleneck_validation.npy"
FLATTENED_TOP_MODEL_PATH = "auxiliary/resnet_bottleneck_flattened_fc_model.h5"
POOLED_TOP_MODEL_PATH = "auxiliary/resnet_bottleneck_pooled_fc_model.h5"

TRAIN_CLASS_SIZE = 249
VALIDATION_CLASS_SIZE = 250

IMG_WIDTH, IMG_HEIGHT = 150, 150


# %%
def get_batches(
    path,
    datagen=ImageDataGenerator(preprocessing_function=resnet.preprocess_input),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=16,
    class_mode=None,
    shuffle=False,
    save_to_dir=None,
):
    generator = datagen.flow_from_directory(
        path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=shuffle,
        save_to_dir=save_to_dir,
    )
    return generator

# %%
resnet_model = ResNet50(include_top=False, weights="imagenet")

train_generator = get_batches(
    TRAIN_DATA_DIR, batch_size=TRAIN_CLASS_SIZE, save_to_dir="auxiliary/preview_train"
)
bottleneck_features_train = resnet_model.predict_generator(train_generator, steps=2)
np.save(open(BOTTLENECK_TRAIN_PATH, "wb"), bottleneck_features_train)

validation_generator = get_batches(
    VALIDATION_DATA_DIR,
    batch_size=VALIDATION_CLASS_SIZE,
    save_to_dir="auxiliary/preview_validation",
)
bottleneck_features_validation = resnet_model.predict_generator(validation_generator, steps=2)
np.save(open(BOTTLENECK_VALIDATION_PATH, "wb"), bottleneck_features_validation)

# %%
train_data = np.load(open(BOTTLENECK_TRAIN_PATH, "rb"))
train_labels = np.array([0] * TRAIN_CLASS_SIZE + [1] * TRAIN_CLASS_SIZE)

validation_data = np.load(open(BOTTLENECK_VALIDATION_PATH, "rb"))
validation_labels = np.array([0] * VALIDATION_CLASS_SIZE + [1] * VALIDATION_CLASS_SIZE)

# %% [markdown]
# ## ResNet 4D output flattened

# %%
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer=RMSprop(1e-4), loss="binary_crossentropy", metrics=["accuracy"])

checkpoint = ModelCheckpoint(
    FLATTENED_TOP_MODEL_PATH,
    monitor="val_acc",
    verbose=1,
    save_best_only=True,
    mode="max",
)

model.fit(
    train_data,
    train_labels,
    epochs=100,
    validation_data=(validation_data, validation_labels),
    callbacks=[checkpoint],
)

# %% [markdown]
# ## ResNet 4D output pooled

# %%
model = Sequential()
# model.add(GlobalAveragePooling2D())
model.add(GlobalMaxPooling2D())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer=RMSprop(1e-4), loss="binary_crossentropy", metrics=["accuracy"])

checkpoint = ModelCheckpoint(
    POOLED_TOP_MODEL_PATH, monitor="val_acc", verbose=1, save_best_only=True, mode="max"
)

model.fit(
    train_data,
    train_labels,
    epochs=100,
    validation_data=(validation_data, validation_labels),
    callbacks=[checkpoint],
)

# %% [markdown]
# ## Results

# %% [markdown]
# - Best checkpointed accuracy with Flattening - ~91%
# - With Global Average Pooling - ~89%
# - With Global Max Pooling - ~91%
#
# Models trained on top of ResNet50 significantly outperformed those we trained on top VGG16.
