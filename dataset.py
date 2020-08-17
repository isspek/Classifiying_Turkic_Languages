from pathlib import Path

import numpy as np
import pandas as pd

from logger import logger
from cli import args

DATA_DIR = Path("data")
RAW_DATA = DATA_DIR / "old-newspaper.tsv"

ID2STR = {0: "turkish", 1: "azerbaijan", 2: "uzbek"}
STR2ID = {"turkish": 0, "azerbaijan": 1, "uzbek": 2}


def extract_turkic_texts():
    '''
    This function extract turkic texts and save as seperate tsv files
    :return:
    :rtype:
    '''

    data = pd.read_csv(RAW_DATA, sep="\t")

    for lang in STR2ID.keys():
        lang_data = data[data["Language"].str.lower() == lang]
        lang_data.to_csv(DATA_DIR / "{}.tsv".format(lang), sep="\t", index=False)
        logger.info("{} is extracted.".format(lang))


def stats():
    for lang in STR2ID.keys():
        data = pd.read_csv(DATA_DIR / "{}.tsv".format(lang), sep="\t")
        logger.info("{lang}: {sum} samples".format(lang=lang, sum=len(data)))


def split(random_seed: int):
    '''
    This function split data into train, dev, test as %80, %10, %10 respectively and saved under DATA_DIR / random_seed
    :return:
    :rtype:
    '''
    X_train = []
    X_dev = []
    X_test = []
    y_train = []
    y_dev = []
    y_test = []
    for lang in STR2ID.keys():
        data = pd.read_csv(DATA_DIR / "{}.tsv".format(lang), sep="\t")

        # we don't process all samples due to the storage limitation
        data = data.sample(n=10000, random_state=0)
        texts = data["Text"].to_numpy()

        indices = np.arange(len(texts))
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        texts = texts[indices]

        train_size = int(0.8 * len(texts))
        dev_size = int(0.1 * len(texts))

        train_texts = texts[:train_size]
        dev_texts = texts[train_size:train_size + dev_size]
        test_texts = texts[train_size + dev_size:]

        assert len(texts) == len(train_texts) + len(dev_texts) + len(test_texts)

        for text in train_texts:
            X_train.append(text)
            y_train.append(STR2ID[lang])

        for text in dev_texts:
            X_dev.append(text)
            y_dev.append(STR2ID[lang])

        for text in test_texts:
            X_test.append(text)
            y_test.append(STR2ID[lang])

    splits_dir = DATA_DIR / str(random_seed)

    splits_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Recording splits")
    np.savez_compressed(splits_dir / "data.npz", X_train=np.asarray(X_train).reshape(-1, 1)
             , X_dev=np.asarray(X_dev).reshape(-1, 1),
             X_test=np.asarray(X_test).reshape(-1, 1),
             y_train=np.asarray(y_train).reshape(-1, 1),
             y_dev=np.asarray(y_dev).reshape(-1, 1),
             y_test=np.asarray(y_test).reshape(-1, 1))


def read_splits(random_seed):
    splits_dir = DATA_DIR / str(random_seed)
    data = np.load(splits_dir / "data.npz")
    X_train = data["X_train"]
    X_dev = data["X_dev"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_dev = data["y_dev"]
    y_test = data["y_test"]
    return X_train, y_train, X_dev, y_dev, X_test, y_test


if __name__ == '__main__':
    if args.mode == "extract_turkic_texts":
        extract_turkic_texts()
    elif args.mode == "stats":
        stats()
    elif args.mode == "split":
        split(args.random_seed)
