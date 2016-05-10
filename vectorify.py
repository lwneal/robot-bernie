"""
Usage:
        vectorify.py --dictionary <glove.txt> --input <corpus.txt>

Arguments:
        -d, --dictionary <glove.txt>    A pretrained word vector model
        -i, --input <corpus.txt>        Input text to train on
"""
import re
import sys
import docopt
import numpy as np
import random


def parse_dictionary(lines):
    word_vectors = {}
    i = 0
    sys.stdout.write("\n")
    for line in lines:
        tokens = line.strip().split()
        if len(tokens) < 2:
            print("Warning, error parsing word vector on line: {}".format(line))
            continue
        word = tokens[0]
        vector = np.asarray([[float(n) for n in tokens[1:]]])[0]
        word_vectors[word] = vector
        if i % 1000 == 0:
            sys.stdout.write("\rProcessed {}/{} ({:.01f} percent)    ".format(i, len(lines), 100.0 * i / len(lines)))
            sys.stdout.flush()
        i += 1
    sys.stdout.write("\n")

    print("Finished parsing {} words".format(len(word_vectors)))
    return word_vectors


def closest_word(word2vec, unknown_vector):
    best_distance = 1000
    best_word = '?'
    best_vector = unknown_vector
    for word, vector in word2vec.items():
        distance = np.linalg.norm(unknown_vector - vector)
        if distance < best_distance:
            best_distance = distance
            best_word = word
            best_vector = vector
    return best_word, best_vector


def train_model(wordvectors, maxlen):
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Dropout
    from keras.layers.recurrent import GRU
    word_count, dimensionality = wordvectors.shape
    print('Compiling model...')
    model = Sequential()
    model.add(GRU(256, input_shape=(maxlen, dimensionality)))
    model.add(Dropout(0.2))
    model.add(Dense(dimensionality))
    model.add(Activation('tanh'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    print('Finished compiling model')

    # cut the text in semi-redundant sequences of maxlen characters
    step = 1
    sentences = []
    next_words = []
    for i in range(0, len(wordvectors) - maxlen, step):
        sentences.append(wordvectors[i: i + maxlen])
        next_words.append(wordvectors[i + maxlen])
    print('Training on {} sequences'.format(len(sentences)))

    X = np.asarray(sentences)
    y = np.asarray(next_words)

    for iteration in range(1, 1000):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X, y, batch_size=256, nb_epoch=6, verbose=1)
        yield model



def main(dict_file, corpus_file):
    print("Loading dictionary {}".format(dict_file))
    lines = open(dict_file).readlines()
    print("Loaded {} lines".format(len(lines)))
    word2vec = parse_dictionary(lines)

    print("Converting training data {} to word vectors...".format(corpus_file))
    text = open(corpus_file).read()
    text = re.sub(r'[^a-zA-Z0-9\.]+', ' ', text).lower().replace('.', ' . ')
    words = text.split()
    print("Vectorized {} words".format(len(words)))
    wordvectors = np.asarray([word2vec.get(word) for word in words if word in word2vec])

    maxlen = 4
    iter = 0
    for model in train_model(wordvectors, maxlen):
        iter += 1
        print("Trained model iteration {}".format(iter))

        idx = random.randint(0, wordvectors.shape[0] - maxlen)
        x = wordvectors[idx:idx + maxlen]

        sys.stdout.write(' '.join([closest_word(word2vec, v)[0] for v in x]))
        for i in range(maxlen):
            y = model.predict(np.asarray([x]))[0]
            predicted_word, predicted_vector = closest_word(word2vec, y)
            sys.stdout.write(" {}".format(predicted_word))
            sys.stdout.flush()
            distance = np.linalg.norm(word2vec[predicted_word] - y)
            y_column = predicted_vector.reshape((1, -1))
            x = np.concatenate( (x[1:], y_column) )
        sys.stdout.write('\n')
    print("Finished run with NO normalization of input vectors, maxlen {}, GRU single layer".format(maxlen))
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)
    dict_file = arguments['--dictionary']
    corpus_file = arguments['--input']
    main(dict_file, corpus_file)
