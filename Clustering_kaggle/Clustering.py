# coding: utf-8
from collections import defaultdict

import pandas as pd
from typing import List
from spacy.lang.ru.stop_words import STOP_WORDS
from sklearn import cluster
from sklearn.feature_extraction.text import TfidfVectorizer


STOP_WORDS = list(STOP_WORDS)


def process_text(str):
    """ Converting a string of pre-lemmatized words into a list of tokens """
    return [s for s in str.split() if not s.isspace()]


class TextsPairClassifier(object):

    def __init__(self, data: List[str]):
        self.pair_labels = defaultdict(lambda: 0)
        vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, tokenizer=process_text, min_df=3, max_df=0.7)

        term_doc_matrix = vectorizer.fit_transform(data)

        clusterizer = cluster.Birch(n_clusters=6, branching_factor=40, threshold=0.5)

        clusterizer.fit(term_doc_matrix.toarray())

        # saving the predicted cluster labels
        self.pair_labels = clusterizer.labels_

    def label(self, id1: int, id2: int):
        if self.pair_labels[id1] == self.pair_labels[id2]:
            return 1
        else:
            return 0


def generate_submission():

    # reading data
    texts = pd.read_csv("normalized_texts.csv", index_col="id", encoding="utf-8")
    pairs = pd.read_csv("pairs.csv", index_col="id")

    # preparing clusters on object creation and initialization
    classifier = TextsPairClassifier(texts["paragraph_lemmatized"].to_list())

    # generating submission
    with open("submission.csv", "w", encoding="utf-8") as output:
        output.write("id,gold\n")
        for index, id1, id2 in pairs.itertuples():
            result = classifier.label(id1-1, id2-1)
            output.write("%s,%s\n" % (index, result))


if __name__ == "__main__":
    generate_submission()
