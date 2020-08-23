import numpy as np
import pandas as pd

from cli import get_args
from dataset import STR2ID, DATA_DIR, read_splits
from feature_extractor import FeatureExtractor
from logger import logger


def stats(type):
    if type == "all":
        for lang in STR2ID.keys():
            data = pd.read_csv(DATA_DIR / "{}.tsv".format(lang), sep="\t")
            logger.info("{lang}: {sum} samples".format(lang=lang, sum=len(data)))
    elif type == "experimental":
        random_splits = [42, 1, 15]
        for i in random_splits:
            logger.info("Random seed {}".format(i))
            X_train, y_train, X_dev, y_dev, X_test, y_test = read_splits(i)

            logger.info("Train {}".format(len(y_train)))
            logger.info("Dev {}".format(len(y_dev)))
            logger.info("Test {}".format(len(y_test)))

            feat_extractor = FeatureExtractor()

            X_sents = [feat_extractor.extract_sents(text[0]) for text in X_train]
            X_words = [feat_extractor.extract_words(text) for text in X_train]

            X_sent_count = [len(sent) for sent in X_sents]  # average sentences in post
            X_word_count = [len(words) for words in X_words]  # average words in post
            logger.info(np.mean(X_sent_count))
            logger.info(np.mean(X_word_count))


if __name__ == '__main__':
    args = get_args()
    stats(args.type)
