from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.datasets.data_utils import get_file
from keras.optimizers import RMSprop
import numpy as np
import sys
import random
import sys
import os
import re
from StringIO import StringIO
from LSTMPeephole import LSTMPeephole

BATCH_SIZE = 256

def read_text_from_file(filename):
    text = open(filename).read()
    text = text.replace('%', ' percent ')
    text = re.sub(r' +', ' ', text).lower()
    text = re.sub(r'\n+', '\n', text).lower()
    text = re.sub(r'[^a-zA-Z0-9\.\n\,\';\- ]+', '', text).lower()
    return text

def make_char_lookup_table(text):
    chars = set(text)
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    return char_indices, indices_char

def generate_text_stream(text, offset=0):
    fp = StringIO(text)
    fp.seek(offset)
    while True:
        val = fp.read(1)
        if not val:
            fp.seek(0)
            continue
        yield val

def onehot_encode(generator, char_indices):
    char_count = len(char_indices)
    for val in generator:
        idx = char_indices[val]
        v = np.zeros(char_count)
        v[idx] = 1
        yield v

def generate_training_data(text, char_indices, batch_size=BATCH_SIZE):
    char_count = len(char_indices)
    X = np.zeros((batch_size, 1, char_count))
    y = np.zeros((batch_size, char_count))

    generators = []
    for i in range(batch_size):
        offset = random.randint(0, len(text))
        g = onehot_encode(generate_text_stream(text, offset), char_indices)
        generators.append(g)

    for i in range(batch_size):
        X[i] = next(generators[i]).reshape(X[i].shape)

    indices_char = {v:k for (k,v) in char_indices.items()}
    
    while True:
        for i in range(batch_size):
            y[i] = next(generators[i])
        yield (X, y)
        X[:,0,:] = y[:]

def build_model(char_count, batch_size=BATCH_SIZE):
    model = Sequential()
    model.add(GRU(512, return_sequences=True, batch_input_shape=(batch_size, 1, char_count), stateful=True))
    model.add(Dropout(0.2))
    model.add(GRU(512, return_sequences=False, stateful=True))
    model.add(Dropout(0.2))
    model.add(Dense(char_count))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def sample(a, temperature=.1):
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


def np_to_char(x, indices_char):
    if not x.any():
        return '?'
    idx = np.nonzero(x)[0][0]
    return indices_char[idx].replace('\n', 'NEWLINE')


def build_visualization(layers, old_weights, run_name, iteration):
    visualization_rows = []
    for i, layer in enumerate(layers):
        weights = layer.get_value()
        print("Weights layer {} max is {} min is {}".format(layer, weights.max(), weights.min()))
        weight_update = np.abs(weights - old_weights[i])
        weight_update *= 255.0 / 0.001
        weights_normalized = weights * 128.0 / weights.max() + 128.0
        visualization_rows.append(np.concatenate((weights_normalized, weight_update), axis=1))
        old_weights[i] = weights
    from PIL import Image
    visualization = np.concatenate(visualization_rows)
    Image.fromarray(visualization).convert('RGB').save('/tmp/visualization.png')
    Image.fromarray(visualization).convert('RGB').save('/tmp/visualization_{}_iter{:04d}.png'.format(run_name, iteration))


def main(run_name, text):
    chars = set(text)
    print('Found {} distinct characters: {}'.format(len(chars), ''.join(chars)))
    char_indices, indices_char = make_char_lookup_table(text)

    model = build_model(char_count=len(char_indices))

    # train the model, output generated text after each iteration
    layers = [layer for layer in model.trainable_weights if len(layer.get_value().shape) > 1 and layer.get_value().shape[1] == 512]
    old_weights = [layer.get_value() for layer in layers]
    generator = generate_training_data(text, char_indices)
    for iteration in range(1, 1000):
        print('-' * 50)
        print('Iteration {}'.format(iteration))
        model.reset_states()

        for i in range(512):
            X, y = next(generator)
            results = model.train_on_batch(X, y)
            sys.stdout.write("\rBatch {} Loss: {}\t".format(i, results))
            sys.stdout.flush()
        sys.stdout.write('\n')

        build_visualization(layers, old_weights, run_name, iteration)
        
        """
        history = model.fit_generator(generator, samples_per_epoch=2**16, nb_epoch=1, verbose=1)
        print history.history
        """

        model.reset_states()
        next_char = random.choice(char_indices.keys())
        for i in range(512):
            next_char = predict(model, next_char, char_indices, indices_char)
            sys.stdout.write(next_char)
            sys.stdout.flush()
        sys.stdout.write('\n')
        # Save model parameters
        model.save_weights('{}.iter{}.h5'.format(run_name, iteration), overwrite=True)


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

