import sys
from collections import OrderedDict
from typing import Callable

from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
import numpy as np
import os
import random

from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression

LIMIT = 10000


class HW4Word2Vec(TransformerMixin):
    def __init__(self, word2vec_model: Word2VecKeyedVectors,
                 weights_supplier: Callable[[np.ndarray], np.ndarray]) -> None:
        self._word2vec_model = word2vec_model
        self._weights_supplier = weights_supplier
        self._zeros_vect = np.zeros(word2vec_model['.'].shape)

    def transform(self, X):
        new_X = []

        for sentence in X:
            new_X.append(self._generate_weighted_sentence_vector(sentence))

        transformed_input = np.array(new_X)
        transformed_input = transformed_input.reshape((transformed_input.shape[0], transformed_input.shape[-1]))
        return transformed_input

    def fit_transform(self, X, y=None, **fit_params):
        transform = self.transform(X)
        return transform

    def fit(self, X, y=None, **fit_params):
        return self

    def _generate_weighted_sentence_vector(self, sentence: str):
        embeding_vectors = np.array([self._word2vec_model[w] for w in sentence.split() if w in self._word2vec_model] or
                                    [self._zeros_vect])
        weights = self._weights_supplier(embeding_vectors)

        result = np.matmul(embeding_vectors.T, weights).T
        total_words = embeding_vectors.shape[0]

        return result / total_words


def read_speaker_sentences(speakre_db_path: str) -> list:
    speaker_samples = []

    for source_file in os.listdir(speakre_db_path):
        with open(os.path.join(speakre_db_path, source_file), 'r', encoding='UTF-8') as f:
            speaker_samples += [s.strip() for s in f.readlines()]

    return speaker_samples


def get_even_samples(input_dir1, input_dir2, limit=None):
    speaker1 = read_speaker_sentences(input_dir1)
    speaker2 = read_speaker_sentences(input_dir2)
    # Down sample ...
    speaker1_sentences, speaker2_sentences = len(speaker1), len(speaker2)

    limit = min(speaker1_sentences, speaker2_sentences) if limit is None else limit

    speaker1 = random.choices(speaker1, k=limit)
    speaker2 = random.choices(speaker2, k=limit)

    speaker1 = np.char.array(speaker1, unicode=True)
    speaker2 = np.char.array(speaker2, unicode=True)

    X = np.concatenate((speaker1, speaker2))
    y = np.zeros((X.shape[0],))
    y[limit:] = 1

    return X, y


def distance_of_10_pairs_of_words(X: np.ndarray, word2vec_model: Word2VecKeyedVectors):
    speaker_offset = int(X.shape[0] / 2)
    total_samples = 10
    speaker_1_random = X[np.random.choice(speaker_offset, total_samples, replace=False)]
    speaker_2_random = X[speaker_offset + np.random.choice(speaker_offset, total_samples, replace=False)]

    for speaker_1, speaker_2 in zip(speaker_1_random, speaker_2_random):
        speaker_1 = [w for w in speaker_1.split() if w in word2vec_model]
        speaker_2 = [w for w in speaker_2.split() if w in word2vec_model]

        distance = word2vec_model.n_similarity(speaker_1, speaker_2)
        print("The distance between:")
        print(" ".join(speaker_1))
        print(" ".join(speaker_2))
        print("Distance: ", distance)


def classify_speakers(X: np.ndarray, y: np.ndarray, word2vec_model: Word2VecKeyedVectors):
    def naive_weights(words_vectors: np.ndarray):
        return np.ones((words_vectors.shape[0], 1), )

    def heavy_prefix(words_vectors: np.ndarray):
        weight = np.ones((words_vectors.shape[0], 1), dtype=np.float) * 0.1
        for i in range(min(3, words_vectors.shape[0])):
            weight[i] = 1

        return weight

    def embedding_mean(words_vector: np.ndarray):
        thresh = 0.8
        mean_vect = words_vector.mean(axis=0)

        normalized_mean_vect = mean_vect / np.linalg.norm(mean_vect)
        normalized_words_vector = (words_vector.T / np.linalg.norm(words_vector, axis=1)).T

        weights = np.dot(normalized_words_vector, normalized_mean_vect)
        weights[weights >= thresh] = 1
        weights[weights < thresh] = 0.1
        return weights

    results = OrderedDict()
    results['a'] = handle_pipeline_weights_algorithm(word2vec_model, X, y, naive_weights)
    results['b'] = handle_pipeline_weights_algorithm(word2vec_model, X, y, heavy_prefix)
    results['c'] = handle_pipeline_weights_algorithm(word2vec_model, X, y, embedding_mean)

    formatted_result = []
    for experiment, accuracy in results.items():
        formatted_result.append("accuracy {}: {:0.2f}%".format(experiment, accuracy))

    return formatted_result


def handle_pipeline_weights_algorithm(word2vec_model, X, y, naive_weights):
    naive_pipeline = make_pipeline(HW4Word2Vec(word2vec_model, naive_weights), LogisticRegression())
    scores = cross_validate(naive_pipeline, X, y, cv=10, return_train_score=False, scoring=['accuracy'])
    return np.average(scores['test_accuracy']) * 100


def main(speaker_1_dir, speaker_2_dir, word2vec, output_file):
    print("Loading speakers")
    X, y = get_even_samples(speaker_1_dir, speaker_2_dir, limit=LIMIT)
    print("Loading word2vec")
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec, binary=False)

    # distance_of_10_pairs_of_words(X, word2vec_model)
    formatted_result = classify_speakers(X, y, word2vec_model)
    formatted_result = "\n".join(formatted_result)
    print(formatted_result)

    with open(output_file, 'w') as fd:
        fd.write(formatted_result)


if __name__ == "__main__":
    main(*sys.argv[1:])
