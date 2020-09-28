import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
#from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

test_file_name = "the_way_of_kings"
FILE_MODEL_NAME = test_file_name + "_model.sav"
DATA_FILE_NAME = test_file_name + ".txt"
TWEETS_FILE_NAME = "tweets_storage.txt"
NR_OF_TRAINING_CHARACTERS = 500000
NR_UNITS = 300

def save_model(filename, model):
    pickle.dump(model, open(filename, 'wb'))


def load_model(filename):
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


def load_data():

    # load text
    filename = DATA_FILE_NAME
    text = (open(filename).read())

    # tweets_file_name = TWEETS_FILE_NAME
    # text = (open(tweets_file_name).read()) + "\n" + text

    text = text[0:NR_OF_TRAINING_CHARACTERS]
    # text = text.lower()
    # print(text)

    # mapping characters with integers
    unique_chars = sorted(list(set(text)))

    char_to_int = {}
    int_to_char = {}

    for i, c in enumerate(unique_chars):
        char_to_int.update({c: i})
        int_to_char.update({i: c})

    # print(char_to_int)
    # print(int_to_char)

    # preparing input and output dataset
    X = []
    Y = []
    number_period = 100

    for i in range(0, len(text) - number_period, 1):
        sequence = text[i:i + number_period]
        label = text[i + number_period]
        X.append([char_to_int[char] for char in sequence])
        Y.append(char_to_int[label])

    # print("X", X)
    # print("Y", Y)

    # reshaping, normalizing and one hot encoding
    X_modified = np.reshape(X, (len(X), number_period, 1))
    # print("X_modified", X_modified)
    X_modified = X_modified / float(len(unique_chars))
    # print("X_modified", X_modified)
    Y_modified = np_utils.to_categorical(Y)
    # print("Y_modified", Y_modified)

    return unique_chars, int_to_char, char_to_int, X, Y, X_modified, Y_modified


def generate_content(model, nr_chars):

    unique_chars, int_to_char, char_to_int, X, Y, X_modified, Y_modified = load_data()

    # picking a random seed
    start_index = np.random.randint(0, len(X) - 2)
    new_string = X[start_index]
    s1 = "this is a test. please give me a better output thi"
    for i in range(0, len(s1)):
        new_string[i] = char_to_int[s1[i]]
    content = ""

    # generating characters
    for i in range(nr_chars):
        x = np.reshape(new_string, (1, len(new_string), 1))
        x = x / float(len(unique_chars))

        # predicting
        pred_index = np.argmax(model.predict(x, verbose=0))
        char_out = int_to_char[pred_index]
        print("char_out ", char_out)
        # seq_in = [int_to_char[value] for value in new_string]
        # print("seq_in ", seq_in)

        new_string.append(pred_index)
        new_string = new_string[1:len(new_string)]

        content += char_out

    return content


def train_model(X_modified, Y_modified, load_checkpoint=False):

    # defining the LSTM model
    model = Sequential()
    model.add(LSTM(NR_UNITS, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(NR_UNITS))
    model.add(Dropout(0.2))
    model.add(Dense(Y_modified.shape[1], activation='softmax'))

    # Checkpoint loading
    checkpoint_path = test_file_name + "_weights_improvement.hdf5"
    if load_checkpoint:
        model.load_weights(checkpoint_path)

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    epoch_number = 16

    # Checkpointing
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=False)

    # Early Stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10)

    callbacks_list = [checkpoint, early_stop]

    # Create model's image
    #+
    # plot_model(model, to_file=test_file_name + "_model.png", show_shapes=True, show_layer_names=True)

    # fitting the model
    history = model.fit(X_modified, Y_modified, epochs=epoch_number, batch_size=16, validation_split=0.33, callbacks=callbacks_list)

    filename = FILE_MODEL_NAME
    save_model(filename, model)

    # Plot error images

    # Get history losses
    history_dict = history.history
    train_error = history_dict['loss']
    val_error = history_dict['val_loss']
    # Reconverting to currency
    train_errors = train_error#[number ** 0.5 * std_Ys for number in train_error]
    val_errors = val_error#[number ** 0.5 * std_Ys for number in val_error]

    # Plot errors
    f, ax = plt.subplots(1)
    ax.plot(train_errors, 'b', label='Training loss')
    ax.plot(val_errors, 'r', label='Validation loss')
    ax.set_ylim(ymin=0)
    f.set_size_inches(18.5, 10.5)
    ax.legend()
    plt.show()
    f.savefig(test_file_name + '_errors.jpg')
