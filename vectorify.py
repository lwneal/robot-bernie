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
from sklearn.preprocessing import normalize
from train_model import train_model


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
        vector = normalize(np.asarray([[float(n) for n in tokens[1:]]]))[0]
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
    for word, vector in word2vec.items():
        distance = np.linalg.norm(unknown_vector - vector)
        if distance < best_distance:
            best_distance = distance
            best_word = word
    return best_word


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

    iter = 0
    for model in train_model(wordvectors):
        iter += 1
        print("Trained model iteration {}".format(iter))

        maxlen = 16
        idx = random.randint(0, wordvectors.shape[0] - maxlen)
        x = wordvectors[idx:idx + maxlen]

        print("Input: {}".format(' '.join([closest_word(word2vec, v) for v in x])))
        for i in range(10):
            y = model.predict(np.asarray([x]))[0]
            predicted_word = closest_word(word2vec, y)
            distance = np.linalg.norm(word2vec[predicted_word] - y)
            print "Output: {} (distance {})".format(predicted_word, distance)
            y_column = y.reshape((1, -1))
            x = np.concatenate( (x[1:], y_column) )


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)
    dict_file = arguments['--dictionary']
    corpus_file = arguments['--input']
    main(dict_file, corpus_file)
