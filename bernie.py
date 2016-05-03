"""
    A robot Bernie Sanders
    Requirements:
        pip install keras
"""
import numpy as np
import random
import sys
import traceback
import re
import train_bernie
from train_bernie import build_model, predict, make_char_lookup_table

MODEL_FILENAME = 'model.bernie_shuffle.iter999.h5'
TEXT_FILENAME = 'bernie_corpus.txt'

def main():
    text = train_bernie.read_text_from_file(TEXT_FILENAME)
    char_indices, indices_char = make_char_lookup_table(text)
    model = load_model(char_indices)
    print("I'm Senator Bernie Sanders. What's your name?")
    while True:
        try:
            question = raw_input('> ')
            print(ask_bernie(model, question, char_indices, indices_char))
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            print('Error, Entering debugger')
            import pdb; pdb.set_trace()


def load_model(char_indices):
    char_count = len(char_indices)
    model = build_model(char_count)

    print('Decompressing weights...')
    print('Loading weights into model...')
    model.load_weights(MODEL_FILENAME)
    print('Finished loading weights.')
    return model


def ask_bernie(model, question, char_indices, indices_char):
    sentence = re.sub(r'\W+', ' ', question).lower()

    sys.stdout.write("Output: ")
    for c in sentence:
        predict(model, c, char_indices, indices_char)
        sys.stdout.write(c)
        sys.stdout.flush()

    characters = []
    for i in range(512):
        c = predict(model, c, char_indices, indices_char)
        sys.stdout.write(c)
        sys.stdout.flush()
        characters.append(c)
    sys.stdout.write('\n')
    model.reset_states()

    generated = ''.join(characters)
    # Strip off last word and capitalize sentences
    generated = ' '.join(generated.split()[:-1])
    generated = '. '.join([s.strip().capitalize() for s in generated.split('.')]) + '.'
    return generated


if __name__ == '__main__':
    main()
