import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

# Directories
DATA_DIR = "data"
CHECKPOINT_DIR = f"{DATA_DIR}/training_checkpoints"
# File names
TEST_FILE_NAME = "the_way_of_kings"
FILE_MODEL_NAME = f"{DATA_DIR}/{TEST_FILE_NAME}_model.sav"
DATA_FILE_NAME = f"{DATA_DIR}/{TEST_FILE_NAME}.txt"
TWEETS_FILE_NAME = f"{DATA_DIR}/tweets_storage.txt"
ERRORS_PLOT_IMAGE_NAME = f"{DATA_DIR}/{TEST_FILE_NAME}_errors.jpg"
CHECKPOINT_TO_IMPORT = f"{CHECKPOINT_DIR}/ckpt_1"
# Training variables
NR_OF_TRAINING_CHARACTERS = 500000
NR_UNITS = 256
DROPOUT_RATE = 0.2
EPOCH_NUMBER = 64
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.33
LOADING_SEQUENCE_LENGTH = 100


def save_model(filename, model):
    pickle.dump(model, open(filename, 'wb'))


def load_model(filename):
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


def load_data():

    # load text
    text = (open(DATA_FILE_NAME).read())

    # USE TWEETS?
    # tweets_file_name = TWEETS_FILE_NAME
    # text = (open(tweets_file_name).read()) + "\n" + text

    raw_text = text[0:NR_OF_TRAINING_CHARACTERS]
    # text = text.lower()
    # print(text)

    # create mapping of unique chars to integers, and a reverse mapping
    unique_chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(unique_chars))
    int_to_char = dict((i, c) for i, c in enumerate(unique_chars))

    n_chars = len(raw_text)
    n_vocab = len(unique_chars)
    print("Total Characters: ", n_chars)
    print("Total Vocab: ", n_vocab)

    # prepare the dataset of input to output pairs encoded as integers
    dataX = []
    dataY = []

    for i in range(0, n_chars - LOADING_SEQUENCE_LENGTH, 1):
        seq_in = text[i:i + LOADING_SEQUENCE_LENGTH]
        seq_out = text[i + LOADING_SEQUENCE_LENGTH]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])

    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)
    # reshaping, normalizing and one hot encoding
    x_modified = np.reshape(dataX, (n_patterns, LOADING_SEQUENCE_LENGTH, 1))
    # print("x_modified", x_modified)
    x_modified = x_modified / float(n_vocab)
    # print("x_modified", x_modified)
    y_modified = np_utils.to_categorical(dataY)
    # print("y_modified", y_modified)

    return unique_chars, int_to_char, char_to_int, dataX, dataY, x_modified, y_modified


def generate_content(model, nr_chars):

    unique_chars, int_to_char, char_to_int, dataX, dataY, x_modified, y_modified = load_data()

    # Pick a random seed
    start = np.random.randint(0, len(dataX) - 1)
    content_indices = dataX[start]
    content = ""
    print("Seed:")
    print("\"", ''.join([int_to_char[value] for value in content_indices]), "\"")

    # generating characters
    for i in range(nr_chars):
        x = np.reshape(content_indices, (1, len(content_indices), 1))
        x = x / float(len(unique_chars))

        # predicting
        pred_index = np.argmax(model.predict(x, verbose=0))
        char_out = int_to_char[pred_index]
        print("Char out: ", char_out)
        # seq_in = [int_to_char[value] for value in new_string]
        # print("seq_in ", seq_in)

        content_indices.append(pred_index)
        content_indices = content_indices[1:len(content_indices)]

        content += char_out

    return content


def get_history_errors(history):
    history_dict = history.history
    train_error = history_dict['loss']
    val_error = history_dict['val_loss']
    # Reconverting to currency
    train_errors = train_error  # [number ** 0.5 * std_Ys for number in train_error]
    val_errors = val_error  # [number ** 0.5 * std_Ys for number in val_error]
    return train_errors, val_errors


def plot_errors(train_errors, val_errors):
    f, ax = plt.subplots(1)
    ax.plot(train_errors, 'b', label='Training loss')
    ax.plot(val_errors, 'r', label='Validation loss')
    ax.set_ylim(ymin=0)
    f.set_size_inches(18.5, 10.5)
    ax.legend()
    # plt.show()
    f.savefig(ERRORS_PLOT_IMAGE_NAME)


def train_model(x_modified, y_modified, load_checkpoint=False):

    # defining the LSTM model
    model = Sequential()
    model.add(LSTM(NR_UNITS, input_shape=(x_modified.shape[1], x_modified.shape[2]), return_sequences=True))
    model.add(Dropout(DROPOUT_RATE))
    model.add(LSTM(NR_UNITS))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(y_modified.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Checkpoint loading
    if load_checkpoint:
        model.load_weights(CHECKPOINT_TO_IMPORT)

    # Checkpointing
    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt_{epoch}")
    checkpoint = ModelCheckpoint(checkpoint_prefix, monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False)

    # Early Stopping
    early_stop = EarlyStopping(monitor='loss', patience=10)

    # Fitting the model
    history = model.fit(x_modified, y_modified, epochs=EPOCH_NUMBER, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, callbacks=[checkpoint, early_stop])

    # Save latest model
    save_model(FILE_MODEL_NAME, model)

    # Get history losses and plot errors
    train_errors, val_errors = get_history_errors(history)
    # Plot errors
    plot_errors(train_errors, val_errors)

