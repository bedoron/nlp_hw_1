import os
import random
import re
import sys
from typing import List

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from abc import ABCMeta, abstractmethod

# Set this parameter to None to load the maximal amount of sentences, otherwise, LIMIT amount of sentences will be
# loaded per speaker
LIMIT = 10000
K_BEST = 50


class HW3FeatureFinder(TransformerMixin, metaclass=ABCMeta):
    def transform(self, X):
        new_X = []

        for sentence in X:
            new_X.append(self._generate_feature_vector(sentence))

        return np.array(new_X)

    def fit_transform(self, X, y=None, **fit_params):
        transform = self.transform(X)
        return transform

    def fit(self, X, y=None, **fit_params):
        # transform = self.transform(X)
        return self

    @abstractmethod
    def _generate_feature_vector(self, sentence: str):
        pass


class TransformFeatureVector(HW3FeatureFinder):
    def _generate_feature_vector(self, sentence: str):
        return [
            1 if 'הסתייגות' in sentence else 0,
            1 if 'כבוד' in sentence else 0,
            1 if 'רבותי' in sentence else 0,  # The joke about rivlin from eretz nehederet
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


def read_speaker_sentences(speakre_db_path: str) -> list:
    speaker_samples = []

    for source_file in os.listdir(speakre_db_path):
        with open(os.path.join(speakre_db_path, source_file), 'r', encoding='UTF-8') as f:
            speaker_samples += [s.strip() for s in f.readlines()]

    return speaker_samples


def handle_pipeline(X: np.ndarray, y: np.ndarray, pipeline: Pipeline):
    names = {'multinomialnb': 'Naive Bayes', 'decisiontreeclassifier': 'DecisionTree', 'svm': 'SVM',
             'kneighborsclassifier': 'KNN'}
    # scoring = ['precision', 'f1', 'accuracy']
    scoring = ['accuracy']
    classifier_name = names[pipeline.steps[-1][0]]
    scores = cross_validate(pipeline, X, y, cv=10, return_train_score=False, scoring=scoring)

    for score_type in scoring:
        score_entry = 'test_{}'.format(score_type)
        print("- {}: {:0.2f}%".format(classifier_name, np.average(scores[score_entry])*100))

    # score_time and fit_time were left out
    return scores


def question_one_two(top_100_file_path, X, y):
    print("step 1 (my features):\n")
    pipelines = make_pipeline_classifiers(lambda _: [TransformFeatureVector()])
    calculate_all_accuracies(X, y, pipelines)

    print("\nStep2 (top Hebrew words):\n")
    pipelines = make_pipeline_classifiers(lambda _: [TransformHebrewTopFeatureVector(top_100_file_path)])
    calculate_all_accuracies(X, y, pipelines)


def question_three(X: np.ndarray, y: np.ndarray):
    print("\nStep3 (bag-of-words):\n")
    vectorizer = CountVectorizer()
    vectorizer.fit_transform(X.tolist())

    pipelines = make_pipeline_classifiers(lambda _: [vectorizer, TfidfTransformer(use_idf=False)])
    calculate_all_accuracies(X, y, pipelines)


def question_four(X: np.ndarray, y: np.ndarray, output_file_name):
    print("\nStep4 (selected best features):\n")

    vectorizer = CountVectorizer()
    X_as_matrix = vectorizer.fit_transform(X.tolist())
    selector = SelectKBest(k=K_BEST)
    X_new = selector.fit_transform(X_as_matrix, y)
    selected_features = selector.get_support()
    new_vocab = np.asarray(vectorizer.get_feature_names())[selected_features].tolist()

    with open(output_file_name, 'w', encoding="UTF-8") as f:
        f.write("\n".join(new_vocab))

    better_pipeline = make_pipeline_classifiers(lambda _: [CountVectorizer(vocabulary=new_vocab)])
    calculate_all_accuracies(X, y, better_pipeline)


def calculate_all_accuracies(X, y, feature_extractor_pipeline: List):
    for pipeline in feature_extractor_pipeline:
        handle_pipeline(X, y, pipeline)


# The producer makes sure that we don't share data by accident
def make_pipeline_classifiers(stages_producer):
    stages = stages_producer(None)  # Type: list
    pipelines = []
    for clasifier in [MultinomialNB(), DecisionTreeClassifier(), KNeighborsClassifier(), ]:  # SVC(gamma='auto')]:
        stages_list = list(stages)
        stages_list.append(clasifier)
        pipelines.append(make_pipeline(*stages_list))

    return pipelines


def main(args):
    input_dir1, input_dir2, top_hebrew_words, best_words_file_output_path = args[1], args[2], args[3], args[4]

    X, y = get_even_samples(limit=LIMIT, input_dir1=input_dir1, input_dir2=input_dir2)

    question_one_two(top_hebrew_words, X, y)
    question_three(X, y)

    question_four(X, y, best_words_file_output_path)


if __name__ == "__main__":
    main(sys.argv)
