import os
import random
import re

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from abc import ABCMeta, abstractmethod

# Set this parameter to None to load the maximal amount of sentences, otherwise, LIMIT amount of sentences will be
# loaded per speaker
LIMIT = 10000

RVLN = 'rivlin'
ESTN = 'edelstein'

TO = {
    RVLN: 0,
    ESTN: 1
}

FROM = [RVLN, ESTN]

EDELSTEIN_SOURCE_LOCATION = os.path.join('resources', ESTN)
RIVLIN_SOURCE_LOCATION = os.path.join('resources', RVLN)


class HW3FeatureFinder(TransformerMixin, metaclass=ABCMeta):
    def transform(self, X):
        new_X = []

        for sentence in X:
            new_X.append(self._generate_feature_vector(sentence))

        return np.array(new_X)

    def fit_transform(self, X, y=None, **fit_params):
        transform = self.transform(X)
        return transform

    @abstractmethod
    def _generate_feature_vector(self, sentence: str):
        pass


class TransformFeatureVector(HW3FeatureFinder):
    def _generate_feature_vector(self, sentence: str):
        return [
            1 if 'הסתייגות' in sentence else 0,
            1 if 'כבוד' in sentence else 0,
            1 if 'רבותיי' in sentence else 0,  # The joke about rivlin from eretz nehederet
            len(sentence.split()),
            1 if 'תודה' in sentence else 0,
            1 if 'חבר' in sentence else 0,
            1 if 'כנסת' in sentence else 0,
            1 if 'בבקשה' in sentence else 0,
            1 if ':' in sentence else 0,
            len(re.findall('בעד', sentence)),
            len(re.findall('נגד', sentence)),
            len(re.findall('\d+', sentence)),
            len(sentence.split(',')),
            1 if sentence.endswith('?') else 0,
            1 if sentence.endswith('.') else 0,
        ]


class TransformHebrewTopFeatureVector(HW3FeatureFinder):
    def _generate_feature_vector(self, sentence: str):
        sentence_parts = set(sentence.split())
        top_words_in_sentence = self._top_words_set.intersection(sentence_parts)

        feature_vector = np.zeros((len(self._top_words),))
        for top_word in top_words_in_sentence:
            feature_vector[self._word_to_index[top_word]] = 1

        return feature_vector

    def __init__(self, top_words_path: str) -> None:
        with open(top_words_path, 'r', encoding='UTF-8') as f:
            self._top_words = [w.strip() for w in f.readlines()]

        self._top_words_set = set(self._top_words)
        self._word_to_index = {k: v for v, k in enumerate(self._top_words)}


def get_even_samples(rivlin_path, edelstein_path, limit=None):
    edelstein = read_speaker_sentences(edelstein_path)
    rivlin = read_speaker_sentences(rivlin_path)
    # Down sample ...
    edelstein_sentences, rivlin_sentences = len(edelstein), len(rivlin)
    print("Read - Edelstein sentences:", edelstein_sentences, "Rivlin sentences:", rivlin_sentences)
    rivlin = random.choices(rivlin, k=edelstein_sentences)

    rivlin = np.char.array(rivlin, unicode=True)
    edelstein = np.char.array(edelstein, unicode=True)

    if limit and limit < edelstein_sentences:
        print("Limit was set to", limit, "sentences per speaker")
        rivlin = rivlin[:limit]
        edelstein = edelstein[:limit]
        edelstein_sentences = limit
    else:
        print("Using - Edelstein sentences:", edelstein_sentences, "Rivlin sentences:", len(rivlin))

    X = np.concatenate((rivlin, edelstein))
    y = np.zeros((edelstein_sentences * 2,))
    y[edelstein_sentences:] = TO[ESTN]

    return X, y


def read_speaker_sentences(speakre_db_path: str) -> list:
    speaker_samples = []

    for source_file in os.listdir(speakre_db_path):
        with open(os.path.join(speakre_db_path, source_file), 'r', encoding='UTF-8') as f:
            speaker_samples += [s.strip() for s in f.readlines()]

    return speaker_samples


def handle_pipeline(pipeline: Pipeline, X: np.ndarray, y: np.ndarray):
    names = {'multinomialnb': 'Naive Bayes', 'decisiontreeclassifier': 'DecisionTree', 'svm': 'SVM',
             'kneighborsclassifier': 'KNN'}
    # scoring = ['precision', 'f1', 'accuracy']
    scoring = ['accuracy']
    classifier_name = names[pipeline.steps[-1][0]]
    scores = cross_validate(pipeline, X, y, cv=10, return_train_score=False, scoring=scoring)

    for score_type in scoring:
        score_entry = 'test_{}'.format(score_type)
        print("\t%s%s: %0.2f (+/- %0.2f)" % (
            '', classifier_name, scores[score_entry].mean(), scores[score_entry].std() * 2))

    # score_time and fit_time were left out
    return scores


def main():
    X, y = get_even_samples(limit=LIMIT, rivlin_path=RIVLIN_SOURCE_LOCATION, edelstein_path=EDELSTEIN_SOURCE_LOCATION)
    top_100_file_location = os.path.join('resources', 'hebrew top 100.txt')

    # question_one_two(top_100_file_location, X, y)
    question_three(X, y)

    # question_four(X, y)
    print('Whoa!')


def question_one_two(top_100_file_path, X, y):
    print("**** MY FEATURE VECTOR ****")
    tfv = TransformFeatureVector()
    check_scores(tfv, X, y)
    print("**** HEBREW TOP 100 ****")
    thtfv = TransformHebrewTopFeatureVector(top_100_file_path)
    check_scores(thtfv, X, y)


def check_scores(feature_extractor, X: np.ndarray, y: np.ndarray):
    pipeline = make_pipeline(feature_extractor)
    calculate_all_accuracies(pipeline, X, y)


def question_three(X: np.ndarray, y: np.ndarray):
    print("**** Using count vectorizer with tf-idf ****")
    vectorizer = CountVectorizer()
    vectorizer.fit_transform(X.tolist())
    print("Amount of features:", len(vectorizer.get_feature_names()))
    pipeline = make_pipeline(vectorizer, TfidfTransformer(use_idf=False))
    calculate_all_accuracies(X, y, pipeline)


def question_four(X: np.ndarray, y: np.ndarray):
    print("**** Using KBest ****")

    vectorizer = CountVectorizer()
    X_as_matrix = vectorizer.fit_transform(X.tolist())
    selector = SelectKBest(k=50)
    X_new = selector.fit_transform(X_as_matrix, y)
    selected_features = selector.get_support()
    new_vocab = np.asarray(vectorizer.get_feature_names())[selected_features].tolist()

    print("Better vocab:", new_vocab)

    better_pipeline = make_pipeline(CountVectorizer(), SelectKBest(k=50))
    calculate_all_accuracies(X, y, better_pipeline)

    better_pipeline = make_pipeline(CountVectorizer(vocabulary=new_vocab))
    calculate_all_accuracies(X, y, better_pipeline)

    print(X_new.shape)


def calculate_all_accuracies(X, y, feature_extractor_pipeline):
    print("Handling", "->".join([step[0] for step in feature_extractor_pipeline.steps]))

    pipelines = []
    for clasifier in [MultinomialNB(), DecisionTreeClassifier(), KNeighborsClassifier(), ]:  # SVC(gamma='auto')]:
        pipelines += make_pipeline(feature_extractor_pipeline, clasifier),

    for pipeline in pipelines:
        handle_pipeline(pipeline, X, y)


if __name__ == "__main__":
    main()
