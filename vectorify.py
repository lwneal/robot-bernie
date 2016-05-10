"""
Usage:
        vectorify.py --dictionary <glove.txt> --input <corpus.txt> --tag <name>

Arguments:
        -d, --dictionary <glove.txt>    A pretrained word vector model
        -i, --input <corpus.txt>        Input text to train on
        -t, --tag <name>                Name to identify this experiment
"""
import re
import sys
import docopt
import numpy as np
import random
import time


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


def train_model(wordvectors, batch_size):
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Dropout
    from keras.layers.recurrent import GRU
    from keras.optimizers import RMSprop
    word_count, dimensionality = wordvectors.shape
    print('Compiling model...')
    model = Sequential()
    model.add(GRU(512, return_sequences=True, batch_input_shape=(batch_size, 1, dimensionality), stateful=True))
    model.add(Dropout(0.2))
    model.add(GRU(512, return_sequences=False, stateful=True))
    model.add(Dropout(0.2))
    model.add(Dense(dimensionality))
    model.add(Activation('tanh'))
    optimizer = RMSprop(lr=.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    print('Finished compiling model')
    return model


def generate_training_data(wordvectors, batch_size):
    word_count, dimensionality = wordvectors.shape
    indices = [random.randint(1, len(wordvectors) - 1) for i in range(batch_size)]
    while True:
        X = np.array([wordvectors[i] for i in indices]).reshape( (batch_size, 1, dimensionality) )
        y = np.array([wordvectors[i+1] for i in indices])
        yield (X, y)
        for i in range(len(indices)):
            indices[i] += 1
            if indices[i] >= len(wordvectors) - 1:
                indices[i] = 0


def main(dict_file, corpus_file, tag):
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
    word_count, dimensionality = wordvectors.shape

    batch_size = 128
    model = train_model(wordvectors, batch_size)
    generator = generate_training_data(wordvectors, batch_size)

    start_time = time.time()
    for iteration in range(100):
        print("Starting iteration {} after {} seconds".format(iteration, time.time() - start_time))
        batches_per_minute = 2 ** 18 / batch_size 
        for i in range(batches_per_minute):
            X, y = next(generator)
            results = model.train_on_batch(X, y)
            sys.stdout.write("\rBatch {} Loss: {}\t".format(i, results))
            sys.stdout.flush()
        sys.stdout.write('\n')

        input_len = 20
        idx = random.randint(0, wordvectors.shape[0] - input_len)
        context = wordvectors[idx:idx + input_len]

        for word in context:
            in_array = np.zeros( (batch_size, 1, dimensionality) )
            in_array[0] = word
            model.predict(in_array, batch_size=batch_size)[0]

        print "Input: "
        for word in context[-4:]:
            predicted_word, predicted_vector = closest_word(word2vec, word)
            sys.stdout.write(' ' + predicted_word)
            sys.stdout.write('\n')

        print "Output: "
        for i in range(5):
            in_array = np.zeros( (batch_size, 1, dimensionality) )
            in_array[0] = word
            y = model.predict(in_array, batch_size=batch_size)[0]
            predicted_word, predicted_vector = closest_word(word2vec, y)
            sys.stdout.write(" {}".format(predicted_word))
            sys.stdout.flush()
        sys.stdout.write('\n')

        model.save_weights('model.{}.iter{}.h5'.format(tag, iteration))
    print("Finished")


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)
    dict_file = arguments['--dictionary']
    corpus_file = arguments['--input']
    tag = arguments['--tag']
    main(dict_file, corpus_file, tag)
