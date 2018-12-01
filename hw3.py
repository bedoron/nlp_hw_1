import os
import random
import re
from typing import List

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from abc import ABCMeta, abstractmethod

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

        feature_vector = np.zeros((len(self._top_words), ))
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
    print("Edelstein sentences:", edelstein_sentences, "Rivlin sentences:", rivlin_sentences)
    rivlin = random.choices(rivlin, k=edelstein_sentences)
    print("Edelstein sentences:", edelstein_sentences, "Rivlin sentences:", len(rivlin))

    rivlin = np.char.array(rivlin, unicode=True)
    edelstein = np.char.array(edelstein, unicode=True)

    if limit and limit < edelstein_sentences:
        rivlin = rivlin[:limit]
        edelstein = edelstein[:limit]
        edelstein_sentences = limit

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


def handle_pipeline(pipeline: Pipeline, X: np.ndarray, y: np.ndarray, scoring: List[str]):
    classifier_name = pipeline.steps[-1][0]
    print("Cross validating with", classifier_name)
    scores = cross_validate(pipeline, X, y, cv=10, return_train_score=False, scoring=scoring)
    print(classifier_name, "scores:")

    for score_type in scoring:
        score_entry = 'test_{}'.format(score_type)
        print("\t%-10s: %0.2f (+/- %0.2f)" % (score_type, scores[score_entry].mean(), scores[score_entry].std() * 2))

    # score_time and fit_time were left out
    return scores


def main():
    X, y = get_even_samples(limit=None, rivlin_path=RIVLIN_SOURCE_LOCATION, edelstein_path=EDELSTEIN_SOURCE_LOCATION)
    tfv = TransformFeatureVector()
    # check_scores(tfv, X, y)

    thtfv = TransformHebrewTopFeatureVector(os.path.join('resources', 'hebrew top 100.txt'))
    check_scores(thtfv, X, y)

    print('Whoa!')


def check_scores(feature_extractor: HW3FeatureFinder, X: np.ndarray, y: np.ndarray):
    scoring = ['precision', 'f1', 'accuracy']
    pipelines = []
    pipelines += make_pipeline(feature_extractor, MultinomialNB()),
    pipelines += make_pipeline(feature_extractor, DecisionTreeClassifier()),
    pipelines += make_pipeline(feature_extractor, KNeighborsClassifier()),
    # pipelines += make_pipeline(tfv, DictVectorizer(sparse=True), SVC(gamma='auto')),
    for pipeline in pipelines:
        handle_pipeline(pipeline, X, y, scoring)


# def predict_clf(clf: Pipeline):
#     docs_new = ['שובי דובי חבר הכנסת .',
#                 'זה יכול לשמש כחומר לכל מיני מחקרים .',
#                 'זה חבר הכנסת .',
#                 ]
#     print("Predicting...")
#     result = clf.transform(docs_new)
#     predicted = clf.predict(result)
#
#     for doc, category in zip(docs_new, predicted):
#         print("%r => %r" % (doc, FROM[int(category)]))


if __name__ == "__main__":
    main()
