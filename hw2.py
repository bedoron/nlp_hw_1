import json
import math
import os
import pickle
import random
import re
import sys
from collections import defaultdict
import xml.etree as et
from typing import Dict, List
from xml.etree import ElementTree as et
from functools import wraps

SENTENCE_START = '<SOS>'
SENTENCE_END = '<EOS>'


# TODO: Remove caching before submission
def with_cache(func):
    @wraps(func)
    def my_func(*args, **kwargs):
        cache_file = os.path.join('resources', '{}.pcl'.format(func.__name__))
        if os.path.isfile(cache_file):
            with open(cache_file, 'rb') as fd:
                return pickle.load(fd)

        cached_data = func(*args, **kwargs)
        with open(cache_file, 'wb') as fd:
            pickle.dump(cached_data, fd)

        return cached_data

    return my_func


@with_cache
def extract_to_map(speaker_file):
    speaker_to_speeches = {}
    root = et.parse(os.path.abspath(speaker_file)).getroot()
    for doc in root:
        text = doc[1].text
        speaker = doc[0].text
        if text is None:
            print('Speaker', speaker, 'Had nothing to say')
            continue

        # Add start/end of sentence tokens
        text = "\n".join(["{} {} {}".format(SENTENCE_START, word, SENTENCE_END) for word in text.split("\n")])

        speaker_to_speeches[speaker] = text

    return speaker_to_speeches


def build_unigram_model(corpus: List[str]):
    counter = defaultdict(int)
    for token in corpus:
        counter[token] += 1

    model = defaultdict(float)
    for token, freq in counter.items():
        model[token] = freq / len(corpus)

    return model


def calculate_probability(unigrams: Dict[str, int], sentence: str):
    """
    Do some numeric hoop jumping so we wont underflow
    :param unigrams:
    :param sentence:
    :return:
    """
    probabilities = map(lambda token: unigrams.get(token, 0), sentence.split())
    logs = map(lambda probability: math.log(probability) if probability > 0 else 0, probabilities)
    sentence_probability = math.exp(sum(logs))

    return sentence_probability


def sample_word(population, distribution, sentence: List[str]):
    predicted = random.choices(population=population, weights=distribution, k=1)[0]
    if predicted == SENTENCE_START:
        return 0

    sentence.append(predicted)
    return 1 if predicted != SENTENCE_END else -1


def generate_sentence_from_unigram(xgrams: Dict[tuple, float]):
    sentence = []
    population, values = zip(*xgrams.items())
    while True:
        changed = sample_word(population=population, distribution=values, sentence=sentence)
        if changed == -1:
            break

    return " ".join(sentence[2:-1])


# def generate_sentence(population, distribution, length):
#     sentence = []
#     for word in range(length):
#         changed = sample_word(population, distribution, sentence)
#
#         if changed == -1:
#             break
#
#     return " ".join(sentence) if sentence else ''
#
#
# def generate_sentences(population, distribution, length, num_sentences):
#     sentences = []
#     for i in range(num_sentences):
#         sentences.append(generate_sentence(population, distribution, length))
#
#     return sentences


def print_sentences_probabilities(unigrams):
    sentences = [
        'אני חושב שנתנו לך נתונים לא מדויקים .',
        'אני מגיע לכל ההצבעות בכנסת .',
        'תודה רבה .',
        ' גכג שלום גכקא .',
    ]

    for sentence in sentences:
        probability = calculate_probability(unigrams, sentence)
        print('Probability is', probability, 'sentence:', sentence)


def build_ngram_model(tokenized_text_array: List[str], n=2):
    assert n >= 2  # Method supports only bigrams or more
    # Build ngram tuples from corpus
    ngrams = zip(*[tokenized_text_array[sub:] for sub in range(n)])
    ngrams = [tuple(ngram, ) for ngram in ngrams]

    def gen_padded_ngrams():
        for ngram in ngrams:
            yield ngram

            # No padding needed
            if n < 3:
                continue

            # Handle start of sentence -> we synthesize expanding start of sentence (Pad left)
            padded = []
            if ngram[0] == SENTENCE_START:
                padded += [tuple([SENTENCE_START] * off + list(ngram[1:n - off + 1])) for off in range(2, n)]

            # Pad right
            if ngram[-1] == SENTENCE_END:
                padded += [tuple(list(ngram[0:n - off]) + [SENTENCE_END] * off) for off in range(2, n)]

            for padded_ngram in padded:
                yield padded_ngram

    ngram_model = defaultdict(lambda: defaultdict(int))
    # Build model
    for ngram in gen_padded_ngrams():
        apriors = tuple([ngram for ngram in ngram[:-1]])
        posterior = ngram[-1]
        ngram_model[apriors][posterior] += 1

    # Calculate probabilities
    for apriors, posteriors in ngram_model.items():
        appriors_count = float(sum(posteriors.values()))
        for posterior in posteriors.keys():
            posteriors[posterior] /= appriors_count

    return ngram_model


def generate_sentence_from_xgram(xgrams: Dict[tuple, Dict[str, float]], *start_conditions):
    sentence = []
    sentence += start_conditions if start_conditions else [SENTENCE_START]
    gram = len(sentence)
    while True:
        last_token = tuple(sentence[-gram:])
        last_token_xgrams = xgrams[last_token]
        if not last_token_xgrams:
            break

        population, distribution = zip(*last_token_xgrams.items())
        changed = sample_word(population, distribution, sentence)
        if changed == -1:
            break

    return " ".join(sentence[2:-1])


def main(argv):
    print("Reading merged file.")
    speakers_to_speeches = extract_to_map(os.path.join('resources', 'merged.xml'))

    print("Splitting tokens and sanitizing them.")
    corpus = " ".join(speakers_to_speeches.values())
    tokenized_text_array = re.split('\s+', corpus)

    print("Building unigram model.")
    unigram_model = build_unigram_model(tokenized_text_array)
    print("Printint probabilities for inputs")
    print_sentences_probabilities(unigram_model)
    print("Trying to generate sentence from unigram.")
    for _ in range(10):
        print(generate_sentence_from_unigram(unigram_model))

    print("Building bigram model.")
    bigrams_model = build_ngram_model(tokenized_text_array, 2)
    print("Trying to generate sentence from bigram.")
    for _ in range(10):
        print(generate_sentence_from_xgram(bigrams_model, SENTENCE_START))

    print("Building trigram model.")
    trigrams_model = build_ngram_model(tokenized_text_array, 3)
    print("Trying to generate sentence from trigram.")
    for _ in range(10):
        print(generate_sentence_from_xgram(trigrams_model, SENTENCE_START, SENTENCE_START))


if __name__ == "__main__":
    main(sys.argv)
