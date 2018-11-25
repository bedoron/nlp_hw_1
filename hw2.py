import hashlib
import itertools
import logging
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

logging.basicConfig(format='%(message)s')

CHANGE_WATCHDOG_THRESHOLD = 10
SENTENCE_START_TOKEN = '<SOS>'
SENTENCE_END_TOKEN = '<EOS>'

‎
def filter_special_tokens(sentence):
    return re.sub('\b?(' + re.escape(SENTENCE_START_TOKEN) + '|' + re.escape(SENTENCE_END_TOKEN) + ')\b?', '', sentence)


# TODO: Remove caching before submission
def with_cache(func):
    @wraps(func)
    def my_func(*args, **kwargs):
        cache_file_name = '{}.pcl'.format(func.__name__)
        if args and isinstance(args[0], list):
            hash = hashlib.sha256()
            for i in args[0]:
                hash.update(i.encode())
            cache_file_name = hash.hexdigest() + '_' + cache_file_name

        cache_file = os.path.join('resources', cache_file_name)
        if os.path.isfile(cache_file):
            logging.warning('* Reading for "%s" data from cache file "%s" *', func.__name__, cache_file)
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
            logging.warning('Speaker "%s" had nothing to say', speaker)
            continue

        # Add start/end of sentence tokens
        marked_sentences = ["{} {} {}".format(SENTENCE_START_TOKEN, word, SENTENCE_END_TOKEN) for word in
                            text.split("\n")]
        speaker_tokens = itertools.chain.from_iterable(
            map(lambda sentence: re.split("\s+", sentence), marked_sentences))

        l = list(speaker_tokens)
        if not l:
            continue

        speaker_to_speeches[speaker] = l

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

    return filter_special_tokens(" ".join(sentence[2:-1]))


def print_sentences_probabilities(unigrams):
    sentences = [
        'אני חושב שנתנו לך נתונים לא מדויקים .',
        'אני מגיע לכל ההצבעות בכנסת .',
        'תודה רבה .',
        ' גכג שלום גכקא .',
    ]

    sentences = map(lambda sentence: " ".join([SENTENCE_START_TOKEN, sentence, SENTENCE_END_TOKEN]), sentences)

    logging.info("Printint probabilities for inputs")
    for sentence in sentences:
        probability = calculate_probability(unigrams, sentence)
        logging.info("\tProbability is %s, sentence: \"%s\"", probability, sentence)


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
    return filter_special_tokens(sentence)


def generate_models(corpus: List[str]):
    logging.info("Building unigram model.")
    unigram_model = build_unigram_model(corpus)

    logging.info("Building bigram model.")
    bigram_model = build_ngram_model(corpus, 2)

    logging.info("Building trigram model.")
    trigram_model = build_ngram_model(corpus, 3)

    return {'corpus': corpus, 'unigram_model': unigram_model, 'bigram_model': bigram_model,
            'trigram_model': trigram_model}


def generate_sentences(sentences: int, corpus: List[str], unigram_model: Dict[str, float],
                       bigram_model: Dict[tuple, Dict[str, float]],
                       trigram_model: Dict[tuple, Dict[str, float]]):
    logging.info("Trying to generate sentence from unigram.")
    for _ in range(sentences):
        logging.info("\t %s", generate_sentence_from_xgram(unigram_model, corpus))

    logging.info("Trying to generate sentence from bigram.")
    for _ in range(sentences):
        logging.info("\t %s", generate_sentence_from_xgram(bigram_model, corpus))

    logging.info("Trying to generate sentence from trigram.")
    for _ in range(sentences):
        logging.info("\t %s", generate_sentence_from_xgram(trigram_model, corpus))


def main(argv):
    logging.info("Reading merged file.")
    speakers_to_speeches = extract_to_map(os.path.join('resources', 'merged.xml'))

    logging.info("Splitting tokens and sanitizing them.")
    corpus = list(itertools.chain.from_iterable(speakers_to_speeches.values()))

    models_cache = generate_models(corpus)
    print_sentences_probabilities(models_cache['unigram_model'])

    generate_sentences(sentences=10, **models_cache)

    top_speakers = sorted(speakers_to_speeches.items(), key=lambda pair: len(pair[1]), reverse=True)[:5]

    logging.info("Top speakers: ")
    for speaker_stats in map(lambda pair: "{} - {}".format(pair[0], len(pair[1])), top_speakers):
        logging.info("\t%s", speaker_stats)

    for top_speaker, tokens in top_speakers:
        logging.info("**** Generating text for \"%s\" ****", top_speaker)
        models_cache = generate_models(tokens)
        generate_sentences(sentences=3, **models_cache)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    main(sys.argv)
