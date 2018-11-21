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


def sample_word(population, distribution, sentence: List[str]):
    predicted = random.choices(population=population, weights=distribution, k=1)[0]
    if predicted == SENTENCE_START:
        return 0

    sentence.append(predicted)
    return 1 if predicted != SENTENCE_END else -1


def generate_sentence(population, distribution, length):
    sentence = []
    for word in range(length):
        changed = sample_word(population, distribution, sentence)

        if changed == -1:
            break

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


def build_xgram_matrix(tokenized_text_array: List[str], gram=2):
    xgram_mtarix = defaultdict(lambda: defaultdict(int))
    xgram_apprior_count = defaultdict(int)
    for xgram in window(tokenized_text_array, gram):
        key_tuple = tuple(xgram[i] for i in range(len(xgram) - 1))
        xgram_mtarix[key_tuple][str(xgram[-1])] += 1
        xgram_apprior_count[key_tuple] += 1

    return xgram_mtarix, xgram_apprior_count


def build_xgrams(xgram_matrix: Dict[tuple, Dict[str, int]], xgram_apprior_count: Dict[tuple, int]):
    xgram_freq_mtarix = defaultdict(lambda: defaultdict(float))
    for aprior, posteriors in xgram_matrix.items():
        total_apriors = xgram_apprior_count[aprior]
        for posterior, freq in posteriors.items():
            xgram_freq_mtarix[aprior][posterior] = freq / total_apriors

    return xgram_freq_mtarix


def generate_sentence_from_xgram(xgrams: Dict[tuple, Dict[str, float]], *start_conditions):
    sentence = []
    sentence += start_conditions if start_conditions else [SENTENCE_START]
    gram = len(sentence)
    while True:
        last_token = tuple(sentence[-gram + 1:])
        last_token_xgrams = xgrams[last_token]
        if not last_token_xgrams:
            continue

        population, distribution = zip(*last_token_xgrams.items())
        changed = sample_word(population, distribution, sentence)
        if changed == -1:
            break

    return " ".join(sentence)


def generate_sentence_from_trigram(txgrams, bxgrams):
    # Generate starting conditions for trigram out of bgram
    while True:
        bigram_sentence = generate_sentence_from_xgram(bxgrams).split()
        if len(bigram_sentence) > 1:
            break

    sentence = generate_sentence_from_xgram(txgrams, 3, *bigram_sentence[:2])
    # sentence = generate_sentence_from_xgram(txgrams, SENTENCE_START, SENTENCE_START)
    return sentence


def main(argv):
    speakers_to_speeches = extract_to_map(os.path.join('resources', 'merged.xml'))
    token_counter, total_tokens = count_token_freq(speakers_to_speeches)
    unigrams = build_unigrams(token_counter, total_tokens)

    print_sentences_probabilities(unigrams)

    population, distribution = zip(*unigrams.items())

    sentence = generate_sentences(population, distribution, 15, 3)
    print("\t" + "\n\t".join(sentence))

    # Build bigrams
    print("Building bigram matrix")
    corpus = " ".join(speakers_to_speeches.values())
    tokenized_text_array = re.split('\s+', corpus)
    xgramm, xgramac = build_xgram_matrix(tokenized_text_array, gram=2)
    xgrams = build_xgrams(xgramm, xgramac)

    print("Generated from bigram:")
    for _ in range(10):
        generated_from_bigram = generate_sentence_from_xgram(xgrams)
        print("\t" + generated_from_bigram)

    # Build threegrams
    print("Building trigram matrix")
    txgramm, txgramac = build_xgram_matrix(tokenized_text_array, gram=3)
    txgrams = build_xgrams(txgramm, txgramac)

    # Backoff ? smoothing? Which ?
    print("Generated from trigram:")
    for _ in range(10):
        generated_from_trigram = generate_sentence_from_trigram(txgrams, xgrams)
        print("\t" + generated_from_trigram)


if __name__ == "__main__":
    main(sys.argv)
