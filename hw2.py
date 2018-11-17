import json
import math
import os
import pickle
import random
import re
import sys
from collections import defaultdict
import xml.etree as et
from typing import Dict
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


@with_cache
def count_token_freq(speaker_to_speeches: dict):
    corpus = " ".join(speaker_to_speeches.values())
    token_counter = defaultdict(int)

    total_tokens = 0
    for token in re.split('\s+', corpus):
        token = token.strip()
        token_counter[token] += 1
        total_tokens += 1

    token_counter_tuple = list(token_counter.items())
    token_counter_tuple.sort(key=lambda pair: pair[1], reverse=True)

    print(json.dumps(token_counter_tuple, ensure_ascii=False))
    print('Vocabulary size:', len(token_counter_tuple))

    with open('stats.txt', 'w', encoding='UTF-8') as f:
        f.write(json.dumps(token_counter_tuple, ensure_ascii=False, indent=4))

    return token_counter, total_tokens


def build_unigrams(token_counter: Dict[str, int], total_tokens: int):
    return {token: freq / total_tokens for token, freq in token_counter.items()}


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


def generate_sentence(population, distribution, length):
    sentence = []
    for word in range(length):
        predicted = random.choices(population=population, weights=distribution, k=1)[0]
        if predicted == SENTENCE_START:
            continue

        if predicted == SENTENCE_END:
            break

        sentence.append(predicted)

    return " ".join(sentence) if sentence else ''


def generate_sentences(population, distribution, length, num_sentences):
    sentences = []
    for i in range(num_sentences):
        sentences.append(generate_sentence(population, distribution, length))

    return sentences


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


from collections import deque


def window(seq, n=2):
    it = iter(seq)
    win = deque((next(it, None) for _ in range(n)), maxlen=n)
    yield win
    for e in it:
        win.append(e)
        yield win


def build_bigram_matrix(speakers_to_speeches):
    corpus = " ".join(speakers_to_speeches.values())
    tokens = re.split('\s+', corpus)
    bigram_matrix = defaultdict(lambda: defaultdict(int))
    for bigram in window(tokens):
        bigram_matrix[(bigram[0],)][bigram[1]] += 1

    return bigram_matrix


def build_xgram_matrix(speakers_to_speeches, gram=2):
    corpus = " ".join(speakers_to_speeches.values())
    tokens = re.split('\s+', corpus)
    xgram_mtarix = defaultdict(lambda: defaultdict(int))
    total_xgrams = 0
    for xgram in window(tokens, gram):
        key_tuple = tuple(xgram[i] for i in range(len(xgram) - 1))
        xgram_mtarix[key_tuple][str(xgram[-1])] += 1
        total_xgrams += 1

    return xgram_mtarix, total_xgrams


def build_xgrams(xgram_matrix: Dict[tuple, Dict[str, int]], total_tokens: int):
    xgram_freq_mtarix = defaultdict(lambda: defaultdict(float))
    for aprior, posteriors in xgram_matrix.items():
        for posterior, freq in posteriors.items():
            xgram_freq_mtarix[aprior][posterior] = freq / total_tokens # WRONG! should be line sum (?!)

    return xgram_freq_mtarix


def main(argv):
    speakers_to_speeches = extract_to_map(os.path.join('resources', 'merged.xml'))
    token_counter, total_tokens = count_token_freq(speakers_to_speeches)
    unigrams = build_unigrams(token_counter, total_tokens)

    print_sentences_probabilities(unigrams)

    population, distribution = zip(*unigrams.items())

    sentence = generate_sentences(population, distribution, 15, 3)
    print("\n".join(sentence))

    # Build bigrams
    xgramm, xgramt = build_xgram_matrix(speakers_to_speeches, gram=2)
    xgrams = build_xgrams(xgramm, xgramt)



    for key, val in xgramm.items():
        print("key[", key, "]", val)

    pass


if __name__ == "__main__":
    main(sys.argv)
