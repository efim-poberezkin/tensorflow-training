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
# # Hotdog / Not Hotdog Classification - transfer learning
#
# https://drive.google.com/file/d/1FZ3ZwcPDoEave_xp50Ziue39gMhGPO1W/view

# %% [markdown]
# ## Imports

# %%
from math import ceil

import numpy as np

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

from keras.applications import vgg16
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator

# %% [markdown]
# ## Using the bottleneck features of a pre-trained network

# %%
vgg_model = VGG16(include_top=False, weights="imagenet")
print(vgg_model.summary())

# %%
TRAIN_DATA_DIR = "data/train"
VALIDATION_DATA_DIR = "data/test"
BOTTLENECK_FEATURES_TRAIN_PATH = "auxiliary/bottleneck_features_train.npy"
BOTTLENECK_FEATURES_VALIDATION_PATH = "auxiliary/bottleneck_features_validation.npy"
TOP_MODEL_WEIGHTS_PATH = "auxiliary/bottleneck_fc_model.h5"
TUNED_VGG_WEIGHTS_PATH = "auxiliary/tuned_vgg_model.h5"

TRAIN_CLASS_SIZE = 249
VALIDATION_CLASS_SIZE = 250

IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 16


# %%
def get_data_generator():
    datagen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)
    return datagen


def get_batches(
    path,
    datagen=get_data_generator(),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
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

# %% [markdown]
# We will run our loaded VGG16 model on our training and validation data once, recording the output (the "bottleneck features" from the VGG16 model: the last activation maps before the fully-connected layers) in two numpy arrays. Then we will train a small fully-connected model on top of the stored features.
#
# Below `class_mode=None` means our generator will only yield batches of data without labels, and `shuffle=False` means our data will be in order, so first images will be hotdogs, then not hotdogs. We have exact same number of objects in classes both on train and validation, so here we set `batch_size` to class size and fit generator data in two steps. We do this to use all our training and validation samples by having sample size be multiple of class size, which we wouldn't be able to do if we were using some conventional batch size of 16 or 32 here.

# %%
train_generator = get_batches(
    TRAIN_DATA_DIR,
    batch_size=TRAIN_CLASS_SIZE,
    class_mode=None,
    shuffle=False,
    save_to_dir="auxiliary/preview_train",
)
bottleneck_features_train = vgg_model.predict_generator(train_generator, steps=2)
np.save(open(BOTTLENECK_FEATURES_TRAIN_PATH, "wb"), bottleneck_features_train)

validation_generator = get_batches(
    VALIDATION_DATA_DIR,
    batch_size=VALIDATION_CLASS_SIZE,
    class_mode=None,
    shuffle=False,
    save_to_dir="auxiliary/preview_validation",
)
bottleneck_features_validation = vgg_model.predict_generator(validation_generator, steps=2)
np.save(open(BOTTLENECK_FEATURES_VALIDATION_PATH, "wb"), bottleneck_features_validation)


# %% [markdown]
# We'll create a function for generation of our top fully-connected model since we'll be reusing it later.

# %%
def generate_top_model(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    return model


# %%
train_data = np.load(open(BOTTLENECK_FEATURES_TRAIN_PATH, "rb"))
# the features were saved in order, so recreating the labels is easy
train_labels = np.array([0] * TRAIN_CLASS_SIZE + [1] * TRAIN_CLASS_SIZE)

validation_data = np.load(open(BOTTLENECK_FEATURES_VALIDATION_PATH, "rb"))
validation_labels = np.array([0] * VALIDATION_CLASS_SIZE + [1] * VALIDATION_CLASS_SIZE)

model = generate_top_model(input_shape=train_data.shape[1:])
model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# %%
# es = EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=1)

# reduce_lr = ReduceLROnPlateau(
#     monitor="val_acc", patience=5, verbose=1, factor=0.5, min_lr=1e-4
# )

checkpoint = ModelCheckpoint(
    TOP_MODEL_WEIGHTS_PATH,
    monitor="val_acc",
    verbose=1,
    save_best_only=True,
    mode="max",
)

model.fit(
    train_data,
    train_labels,
    epochs=200,
    batch_size=BATCH_SIZE,
    validation_data=(validation_data, validation_labels),
    callbacks=[checkpoint],
)

# %% [markdown]
# ## Fine-tuning the top layers of a a pre-trained network

# %% [markdown]
# After instantiating the VGG base and loading its weights, we add our previously trained fully-connected classifier on top. We start with a fully-trained classifier, including the top classifier, in order to successfully do fine-tuning. 

# %%
base_model = VGG16(
    include_top=False, weights="imagenet", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
)

top_model = generate_top_model(input_shape=(base_model.output_shape[1:]))
top_model.load_weights(TOP_MODEL_WEIGHTS_PATH)

# add the model on top of the convolutional base
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

# %% [markdown]
# We'll be freezing all convolutional layers up to the last convolutional block and only fine tune the last one which has more specialized features. Let's take a look at our model's layers to freeze the right amount.

# %%
model.layers

# %%
# freeze the first 15 layers (up to the last conv block)
for layer in model.layers[:15]:
    layer.trainable = False

model.compile(
    loss="binary_crossentropy",
    optimizer=RMSprop(lr=3e-6),
    metrics=["accuracy"],
)

# %% [markdown]
# Let's check that the last convolutional block and our top fully-connected model are trainable.

# %%
for layer in model.layers:
    print(f"{layer.name}\t{layer.trainable}")

# %%
train_generator = get_batches(TRAIN_DATA_DIR)
validation_generator = get_batches(VALIDATION_DATA_DIR)

checkpoint = ModelCheckpoint(
    TUNED_VGG_WEIGHTS_PATH,
    monitor="val_acc",
    verbose=1,
    save_best_only=True,
    mode="max",
)

model.fit_generator(
    train_generator,
    steps_per_epoch=ceil(TRAIN_CLASS_SIZE * 2 / BATCH_SIZE),
    epochs=50,
    validation_data=validation_generator,
    validation_steps=ceil(VALIDATION_CLASS_SIZE * 2 / BATCH_SIZE),
    callbacks=[checkpoint],
)

# %% [markdown]
# With transfer learning we were able to increase accuracy to apprx 86% on the validation set comparing to 70% for our custom convnet model.
#
# Comparing fine tuning of the last convolutional block to training only fully connected model on top of VGG16 convolutional base, the increase in accuracy for the best checkpointed model is not that high - around 1%, but the values validation accuracy is varied around during training went from 82-83% to 85-86%.
