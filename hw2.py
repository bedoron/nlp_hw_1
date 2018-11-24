import math
import math
import os
import pickle
import random
import re
import sys
import xml.etree as et
from collections import defaultdict
from functools import wraps
from typing import Dict, List, Union
from xml.etree import ElementTree as et
import logging

logging.basicConfig(format='%(message)s')

CHANGE_WATCHDOG_THRESHOLD = 10
SENTENCE_START_TOKEN = '<SOS>'
SENTENCE_END_TOKEN = '<EOS>'

def FILTER_SPECIAL_TOKENS(sentence):
    return re.sub('\b?(' + re.escape(SENTENCE_START_TOKEN) + '|' + re.escape(SENTENCE_END_TOKEN) + ')\b?', '', sentence)

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
            logging.warning('Speaker "{}" had nothing to say', speaker)
            continue

        # Add start/end of sentence tokens
        text = "\n".join(
            ["{} {} {}".format(SENTENCE_START_TOKEN, word, SENTENCE_END_TOKEN) for word in text.split("\n")])

        speaker_to_speeches[speaker] = text

    return speaker_to_speeches


def build_unigram_model(corpus: List[str]) -> Dict[str, float]:
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
    sentence.append(predicted)

    return predicted == SENTENCE_END_TOKEN


def generate_sentence_from_unigram(xgrams: Dict[tuple, float]):
    sentence = []
    population, values = zip(*xgrams.items())
    generated_end_of_sentence = False
    while not generated_end_of_sentence:
        generated_end_of_sentence = sample_word(population=population, distribution=values, sentence=sentence)

    return FILTER_SPECIAL_TOKENS(" ".join(sentence[2:-1]))


def print_sentences_probabilities(unigrams):
    sentences = [
        'אני חושב שנתנו לך נתונים לא מדויקים .',
        'אני מגיע לכל ההצבעות בכנסת .',
        'תודה רבה .',
        ' גכג שלום גכקא .',
    ]

    for sentence in sentences:
        probability = calculate_probability(unigrams, sentence)
        logging.info("\tProbability is %d, sentence: \"%s\"", probability, sentence)


def build_ngram_model(tokenized_text_array: List[str], n=2):
    assert n >= 2  # Method supports only bigrams or more
    # Build ngram tuples from corpus
    ngrams = zip(*[tokenized_text_array[sub:] for sub in range(n)])
    ngram_model = defaultdict(lambda: defaultdict(int))
    # Build model
    for ngram in ngrams:
        apriors = tuple([ngram for ngram in ngram[:-1]])
        posterior = ngram[-1]
        ngram_model[apriors][posterior] += 1

    # Calculate probabilities
    for apriors, posteriors in ngram_model.items():
        appriors_count = float(sum(posteriors.values()))
        for posterior in posteriors.keys():
            posteriors[posterior] /= appriors_count

    return ngram_model


def generate_markov_chain_seed(tokens: List[str], ngram_size=2):
    seed = random.randint(0, len(tokens) - ngram_size)
    return tokens[seed:seed + ngram_size]


def generate_sentence_from_xgram(xgrams: Union[Dict[tuple, Dict[str, float]], Dict[str, float]], tokens: List[str]):
    first = next(iter(xgrams.keys()))
    ngram_size = len(first) + 1 if isinstance(first, tuple) else 1

    if ngram_size == 1:
        # noinspection PyTypeChecker
        return generate_sentence_from_unigram(xgrams)

    aprior_len, change_watchdog = ngram_size - 1, 0

    # Handle unfortunate random choices where end and start of sentence don't appear one after each other
    while True:
        sentence = generate_markov_chain_seed(tokens, aprior_len)
        if tuple(sentence) in xgrams:
            break

    prev_sentence_len, change_watchdog, generated_sentence_end = aprior_len, 0, False
    while not generated_sentence_end:
        last_aprior = tuple(sentence[-aprior_len:])
        last_token_xgrams = xgrams[last_aprior]
        if not last_token_xgrams:
            break

        population, distribution = zip(*last_token_xgrams.items())
        generated_sentence_end = sample_word(population, distribution, sentence)

    sentence = " ".join(sentence)
    return FILTER_SPECIAL_TOKENS(sentence)


def main(argv):
    logging.info("Reading merged file.")
    speakers_to_speeches \
        = extract_to_map(os.path.join('resources', 'merged.xml'))

    logging.info("Splitting tokens and sanitizing them.")
    corpus = " ".join(speakers_to_speeches.values())
    tokenized_text_array = re.split('\s+', corpus)

    logging.info("Building unigram model.")
    unigram_model = build_unigram_model(tokenized_text_array)
    logging.info("Printint probabilities for inputs")
    print_sentences_probabilities(unigram_model)
    logging.info("Trying to generate sentence from unigram.")
    for _ in range(10):
        logging.info("\t %s", generate_sentence_from_xgram(unigram_model, tokenized_text_array))

    logging.info("Building bigram model.")
    bigrams_model = build_ngram_model(tokenized_text_array, 2)
    logging.info("Trying to generate sentence from bigram.")
    for _ in range(10):
        logging.info("\t %s", generate_sentence_from_xgram(bigrams_model, tokenized_text_array))

    logging.info("Building trigram model.")
    trigrams_model = build_ngram_model(tokenized_text_array, 3)
    logging.info("Trying to generate sentence from trigram.")
    for _ in range(10):
        logging.info("\t %s", generate_sentence_from_xgram(trigrams_model, tokenized_text_array))


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    main(sys.argv)
