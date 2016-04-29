from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
from keras.optimizers import RMSprop
import numpy as np
import sys
import random
import sys
import re

BATCH_SIZE = 128

def read_text_from_file(filename):
    text = open(filename).read()
    text = re.sub(r'[^a-zA-Z0-9\.\n]+', ' ', text).lower()
    return text

def make_char_lookup_table(text):
    chars = set(text)
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    return char_indices, indices_char

def read_training_batch(text, char_indices, batch_size=BATCH_SIZE):
    # Read characters from the text file
    start = random.randint(0, len(text) - batch_size - 1)
    end = start + batch_size + 1
    characters = text[start:end]
    # One-hot encode them
    char_count = len(char_indices)
    X = np.zeros((batch_size, 1, char_count), dtype=np.bool)
    y = np.zeros((batch_size, char_count), dtype=np.bool)
    for i in range(batch_size):
        X[i, 0, char_indices[characters[i]]] = 1
        y[i, char_indices[characters[i + 1]]] = 1
    return X, y

def generate_training_data(text, char_indices, batch_size=BATCH_SIZE):
    X = np.zeros((batch_size, 1, len(char_indices)))
    y = np.zeros((batch_size, len(char_indices)))
    # TODO: 128 streams from different positions in the file, not all the same at once
    for i in range(batch_size):
        X[i, 0, char_indices[text[i]]] = 1
        y[i, char_indices[text[i]]] = 1
    while True:
        for i in range(batch_size, len(text)):
            y = np.roll(y, -1, axis=0)
            y[batch_size - 1, :] = 0
            y[batch_size - 1, char_indices[text[i]]] = 1
            if (i % 1000 == 0):
                print("learning from idx {}".format(i))
            yield (X, y)
            X = y.reshape(X.shape)

def build_model(char_count, batch_size=BATCH_SIZE):
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, batch_input_shape=(batch_size, 1, char_count), stateful=True))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False, stateful=True))
    model.add(Dropout(0.2))
    model.add(Dense(char_count))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def sample(a, temperature=.5):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def predict(model, current_char, char_indices, indices_to_char, batch_size=BATCH_SIZE, temperature=0.1):
    # Ignore all but one value in the batch
    X = np.zeros((batch_size, 1, len(char_indices)))
    X[0, 0, char_indices[current_char]] = 1
    preds = model.predict(X, batch_size=batch_size)[0]
    char_idx = sample(preds, temperature=temperature)
    return indices_to_char[char_idx]


def main(run_name, text):
    chars = set(text)
    print('Found {} distinct characters: {}'.format(len(chars), ''.join(chars)))
    char_indices, indices_char = make_char_lookup_table(text)

    model = build_model(char_count=len(char_indices))

    def np_to_char(x):
        if not x.any():
            return '?'
        idx = np.nonzero(x)[0][0]
        return indices_char[idx]

    def array_to_str(array):
        characters = [np_to_char(x) for x in array]
        return ''.join(characters).replace('\n', '\\N')

    def print_prediction(model, char_indices, indices_char):
        next_char = random.choice(char_indices.keys())
        for i in range(512):
            next_char = predict(model, next_char, char_indices, indices_char)
            sys.stdout.write(next_char)
            sys.stdout.flush()
        sys.stdout.write('\n')

    # train the model, output generated text after each iteration
    generator = generate_training_data(text, char_indices)
    for iteration in range(1, 1000):
        print('-' * 50)
        print('Iteration {}'.format(iteration))
        model.reset_states()
        i = 0
        model.fit_generator(generator, samples_per_epoch=1024 * 1024 * 2, nb_epoch=1)

        """
        # Should work, but results in doubling each character ie. "aabbccddeeffgg..."
        for X, y in generate_training_data(text, char_indices):
            model.train_on_batch(X, y)
            loss = model.test_on_batch(X, y)
            i += 1
            sys.stdout.write('\rProcessed {:.1f}\tLoss: {}\t\t'.format(100.0 * i / len(text), loss))
            sys.stdout.flush()
        sys.stdout.write('\n')
        print 'X: ', array_to_str(X.reshape(X.shape[0], -1))
        print 'y: ', array_to_str(y)
        """

        model.reset_states()
        print_prediction(model, char_indices, indices_char)
        # Save model parameters
        model.save_weights('model.{}.iter{}.h5'.format(run_name, iteration), overwrite=True)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: {} text_corpus.txt run_name'.format(sys.argv[0]))
        print('Text corpus should be at least 100k characters')
        print('It is recommended to run this on a GPU')
        exit()
    filename = sys.argv[1]
    run_name = sys.argv[2]
    text = read_text_from_file(filename)
    print('Text length {} characters'.format(len(text)))
    main(run_name, text)

