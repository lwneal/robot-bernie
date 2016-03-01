"""
    A robot Bernie Sanders
    Requirements:
        pip install keras
"""
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
import numpy as np
import random
import sys
import traceback
import tempfile
import re

MODEL_FILENAME = 'model.bernie.iter45.h5'
char_indices = {'\n': 0, ' ': 1, '.': 2, '1': 3, '0': 4, '3': 5, '2': 6, '5': 7, '4': 8, '7': 9, '6': 10, '9': 11, '8': 12, 'a': 13, 'c': 14, 'b': 15, 'e': 16, 'd': 17, 'g': 18, 'f': 19, 'i': 20, 'h': 21, 'k': 22, 'j': 23, 'm': 24, 'l': 25, 'o': 26, 'n': 27, 'q': 28, 'p': 29, 's': 30, 'r': 31, 'u': 32, 't': 33, 'w': 34, 'v': 35, 'y': 36, 'x': 37, 'z': 38}
maxlen = 20

def main():
    model = load_model()
    print("I'm Senator Bernie Sanders. What's your name?")
    while True:
        try:
            question = raw_input('> ')
            print(ask_bernie(model, question))
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            print('Error, Entering debugger')
            import pdb; pdb.set_trace()


def load_model():
    char_count = len(char_indices)
    # build the model: 2 stacked LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, char_count)))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(char_count))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    print('Decompressing weights...')
    print('Loading weights into model...')
    model.load_weights(MODEL_FILENAME)
    print('Finished loading weights.')
    return model


def ask_bernie(model, question):
    indices_char = {char_indices[key]: key for key in char_indices}
    sentence = question[-maxlen:].lower()
    sentence = re.sub(r'\W+', ' ', sentence)
    if len(sentence) < maxlen:
        sentence = ' ' * (maxlen - len(sentence)) + sentence

    diversity = random.choice([.3, .5, .7, .9])
    generated = sentence
    print()
    for i in range(random.randint(100, 300)):
        x = np.zeros((1, maxlen, len(char_indices)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    # Strip off last word and capitalize sentences
    generated = ' '.join(generated.split()[:-1])
    generated = '. '.join([s.strip().capitalize() for s in generated.split('.')]) + '.'
    return generated


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


if __name__ == '__main__':
    main()
