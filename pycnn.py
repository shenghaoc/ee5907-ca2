# %% [markdown]
# <a href="https://colab.research.google.com/github/shenghaoc/ee5907-ca2/blob/main/cnn.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
import numpy as np
from PIL import Image
from pathlib import Path


# %%
# CONSTANTS
NUM_SUBJECTS = 68
NUM_CHOSEN = 25
NUM_IMAGES_PER_SUBJECT = 170

TRAIN_RATIO = 0.7
NUM_IMAGES = NUM_CHOSEN * NUM_IMAGES_PER_SUBJECT
NUM_TRAIN_IMAGES_PER_SUBJECT = np.int_(np.around(TRAIN_RATIO * NUM_IMAGES_PER_SUBJECT))
NUM_TRAIN_IMAGES = NUM_CHOSEN * NUM_TRAIN_IMAGES_PER_SUBJECT
NUM_TEST_IMAGES = NUM_IMAGES - NUM_TRAIN_IMAGES

NUM_SELFIES = 10
NUM_TRAIN_SELFIES = np.int_(np.around(TRAIN_RATIO * NUM_SELFIES))
NUM_TEST_SELFIES = NUM_SELFIES - NUM_TRAIN_SELFIES
SELFIE_LABEL = NUM_SUBJECTS + 1

NUM_TOTAL_TRAIN_IMAGES = NUM_TRAIN_IMAGES + NUM_TRAIN_SELFIES
NUM_TOTAL_TEST_IMAGES = NUM_TEST_IMAGES + NUM_TEST_SELFIES

SEED1 = 2021
SEED2 = 2022

WIDTH = 32
HEIGHT = 32
NUM_PIXELS = WIDTH * HEIGHT

# New constants due to need to fit input for tensorflow
NUM_PEOPLE = NUM_CHOSEN + 1  # meaning plus the person with 10 selfies
NUM_CHANNELS = 1


# %%
# Ensure that the directory to store figures is created
figures_directory = Path("report") / "figures"
figures_directory.mkdir(exist_ok=True)


# %%
# Must start from 1 to accommodate folder naming scheme
# Choose NUM_CHOSEN elements from NUM_SUBJECTS integers without replacement
chosen = np.random.default_rng(SEED1).choice(
    np.arange(1, NUM_SUBJECTS + 1), NUM_CHOSEN, replace=False
)


# %%
# Load images from disk
# Use lists for manual looping without use of numpy functions
images = []
labels = []

# Assume PIE is in pwd
directory = Path("PIE")
for i in range(len(chosen)):
    # Do not flatten yet, need to split train and test for each subject
    subject_images = []
    subject_labels = []
    subdirectory = directory / str(chosen[i])
    # Order is arbitrary for glob, but better to shuffle anyway
    files = list(subdirectory.glob("*.jpg"))
    np.random.default_rng(SEED2).shuffle(files)
    for filename in files:
        # PIL is slower but OpenCV is unnecessary
        im = Image.open(filename)
        subject_images.append(np.array(im))
        # For tensorflow input, use sequential label
        subject_labels.append(i)
    images.append(subject_images)
    labels.append(subject_labels)


# %%
# Slightly altered code for selfies
selfie_images = []
selfie_labels = []

directory = Path("resized")
# Assume selfies have been resized and folder is in pwd
for filename in directory.glob("*.jpg"):
    im = Image.open(filename)
    selfie_images.append(np.array(im))
    # For tensorflow input, use number of chosen subjects (25) to avoid clashes
    selfie_labels.append(NUM_CHOSEN)


# %%
# Further processing without disk access
# Train-test split
images_train, images_test = np.split(
    np.array(images), [NUM_TRAIN_IMAGES_PER_SUBJECT], axis=1
)
labels_train, labels_test = np.split(
    np.array(labels), [NUM_TRAIN_IMAGES_PER_SUBJECT], axis=1
)

selfie_images_train, selfie_images_test = np.split(
    np.array(selfie_images), [NUM_TRAIN_SELFIES]
)
selfie_labels_train, selfie_labels_test = np.split(
    np.array(selfie_labels), [NUM_TRAIN_SELFIES]
)


# %%
# Flatterning
# For Conv2D, a 4+D tensor is required, add 1 for the grayscale channel
images_train = images_train.reshape(NUM_TRAIN_IMAGES, WIDTH, HEIGHT, NUM_CHANNELS)
selfie_images_train = selfie_images_train.reshape(
    NUM_TRAIN_SELFIES, WIDTH, HEIGHT, NUM_CHANNELS
)
images_test = images_test.reshape(NUM_TEST_IMAGES, WIDTH, HEIGHT, NUM_CHANNELS)
selfie_images_test = selfie_images_test.reshape(
    NUM_TEST_SELFIES, WIDTH, HEIGHT, NUM_CHANNELS
)

labels_train = labels_train.reshape(NUM_TRAIN_IMAGES)
labels_test = labels_test.reshape(NUM_TEST_IMAGES)

# Combine PIE images and selfies
total_images_train = np.append(
    images_train,
    selfie_images_train,
    axis=0,
)
total_labels_train = np.append(labels_train, selfie_labels_train)

total_images_test = np.append(
    images_test,
    selfie_images_test,
    axis=0,
)
total_labels_test = np.append(labels_test, selfie_labels_test)


# %%
# Start of CNN code
import tensorflow as tf
import datetime


# %%
# CONSTANTS
CONV_KERNEL_SIZE = 5
MAX_POOL_KERNEL_SIZE = 2
MAX_POOL_SIZE = 2

BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 100

AUTOTUNE = tf.data.AUTOTUNE


# %%
# Load numpy arrays
# Use built-in one-hot encoder, the numerical labels have no meaning, encoding is necessary to avoid misinterpretation
train_dataset = tf.data.Dataset.from_tensor_slices(
    (total_images_train, tf.keras.utils.to_categorical(total_labels_train))
)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (total_images_test, tf.keras.utils.to_categorical(total_labels_test))
)

train_dataset = (
    train_dataset.cache()
    .shuffle(SHUFFLE_BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=AUTOTUNE)
)
test_dataset = test_dataset.cache().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)


# %%
tf.keras.backend.clear_session()
model = tf.keras.Sequential(
    [
        # Not really necessary, but good practice?
        tf.keras.layers.Rescaling(1.0 / 255, input_shape=(WIDTH, HEIGHT, NUM_CHANNELS)),
        tf.keras.layers.Conv2D(20, CONV_KERNEL_SIZE, activation="relu"),
        tf.keras.layers.MaxPool2D(
            pool_size=(MAX_POOL_KERNEL_SIZE, MAX_POOL_KERNEL_SIZE),
            strides=(MAX_POOL_SIZE, MAX_POOL_SIZE),
        ),
        tf.keras.layers.Conv2D(50, CONV_KERNEL_SIZE, activation="relu"),
        tf.keras.layers.MaxPool2D(
            pool_size=(MAX_POOL_KERNEL_SIZE, MAX_POOL_KERNEL_SIZE),
            strides=(MAX_POOL_SIZE, MAX_POOL_SIZE),
        ),
        tf.keras.layers.Flatten(),  # too many dimensions after Conv2D
        tf.keras.layers.Dense(500, activation="relu"),
        # Keras documentation: often used for last layer because result can be interpreted as
        # a probability distribution
        tf.keras.layers.Dense(NUM_PEOPLE, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(amsgrad=True),  # newest ADAM
    loss=tf.keras.losses.CategoricalCrossentropy(),  # multi-class labeling
    metrics=["accuracy"],
)


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# %%
model.summary()


# %%
# This is cumulative!
model.fit(train_dataset, epochs=100, callbacks=[tensorboard_callback])


# %%
model.predict(test_dataset)


# %%
model.evaluate(test_dataset)



