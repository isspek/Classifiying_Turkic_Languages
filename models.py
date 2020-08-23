import json
import pickle
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from feature_extractor import FeatureExtractor
from logger import logger


class Model(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, test):
        pass


def generate_base_model_name(_config):
    model_name = ''
    for value in _config.values():
        model_name += str(value) + '_'
    return model_name


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion


class Baseline(Model):
    """
    I used the baseline from DSL-2014
    https://github.com/alvations/bayesline-DSL/blob/master/dsl.py
    """

    def __init__(self, _config):
        self.n_min = _config["n_min"]
        self.n_max = _config["n_max"]
        self.max_features = _config["max_features"]
        self.vectorizer_path = 'baseline_vectorizer_multinomialnb_{model_name}.pkl'.format(
            model_name=generate_base_model_name(_config))
        self.input_type = _config["input_type"]
        self.model_dir = None
        self.model_path = 'baseline_{model_name}.pkl'.format(model_name=generate_base_model_name(_config))

    def fit(self, X_train, y_train):
        if self.input_type == "word":
            ngram_vectorizer = CountVectorizer(analyzer='word',
                                               ngram_range=(self.n_min, self.n_max), min_df=1,
                                               max_features=self.max_features)

        elif self.input_type == "char":
            ngram_vectorizer = CountVectorizer(analyzer='char',
                                               ngram_range=(self.n_min, self.n_max), min_df=1,
                                               max_features=self.max_features)

        X_train = ngram_vectorizer.fit_transform(X_train.ravel())

        with open(self.model_dir / self.vectorizer_path, 'wb') as fout:
            pickle.dump(ngram_vectorizer, fout)

        logger.info("Vectorizer is trained.")

        clf = MultinomialNB()
        clf.fit(X_train.todense(), y_train)

        with open(self.model_path, 'wb') as fout:
            pickle.dump(clf, fout)

        logger.info("Model is trained.")

    def predict(self, X_test):
        with open(self.model_dir / self.vectorizer_path, 'rb') as pickle_file:
            ngram_vectorizer = pickle.load(pickle_file)

        with open(self.model_path, 'rb') as pickle_file:
            model = pickle.load(pickle_file)

        X_test = ngram_vectorizer.transform(X_test.ravel())

        preds = model.predict(X_test.todense())

        return preds


class BaselineCombined(Model):

    def __init__(self, _config):
        self.n_min_char = _config["n_min_char"]
        self.n_max_char = _config["n_max_char"]
        self.n_min_word = _config["n_min_word"]
        self.n_max_word = _config["n_max_word"]
        self.max_features = _config["max_features"]
        self.vectorizer_path = 'baseline_vectorizer_multinomialnb_{model_name}.pkl'.format(
            model_name=generate_base_model_name(_config))
        self.input_type = _config["input_type"]
        self.model_dir = None
        self.model_path = 'baseline_{model_name}.pkl'.format(model_name=generate_base_model_name(_config))

    def fit(self, X_train, y_train):
        char_vectorizer = CountVectorizer(analyzer='char',
                                          ngram_range=(self.n_min_char, self.n_max_char), min_df=1,
                                          max_features=self.max_features)
        word_vectorizer = CountVectorizer(analyzer='word',
                                          ngram_range=(self.n_min_word, self.n_max_word), min_df=1,
                                          max_features=self.max_features)

        ngram_vectorizer = FeatureUnion(
            [("char_vectorizer", char_vectorizer), ("word_vectorizer", word_vectorizer)])

        X_train = ngram_vectorizer.fit_transform(X_train.ravel())

        with open(self.model_dir / self.vectorizer_path, 'wb') as fout:
            pickle.dump(ngram_vectorizer, fout)

        logger.info("Vectorizer is trained.")

        clf = MultinomialNB()
        clf.fit(X_train.todense(), y_train)

        with open(self.model_path, 'wb') as fout:
            pickle.dump(clf, fout)

        logger.info("Model is trained.")

    def predict(self, X_test):
        with open(self.model_dir / self.vectorizer_path, 'rb') as pickle_file:
            ngram_vectorizer = pickle.load(pickle_file)

        with open(self.model_path, 'rb') as pickle_file:
            model = pickle.load(pickle_file)

        X_test = ngram_vectorizer.transform(X_test.ravel()).todense()

        preds = model.predict(X_test)

        return preds


class LexiconFeatureBased(Model):
    def __init__(self, _config):
        self.feat_extractor = FeatureExtractor()
        self.params_path = 'feat_based_params_{model_name}.pkl'.format(
            model_name=generate_base_model_name(_config))
        self.model_dir = None
        self.model_path = 'feat_based_{model_name}.pkl'.format(model_name=generate_base_model_name(_config))

    def fit(self, X_train, y_train):
        # ======================= LEXICAL FEATURES ===================================
        X_sents = [self.feat_extractor.extract_sents(text[0]) for text in X_train]
        X_words = [self.feat_extractor.extract_words(text) for text in X_train]

        X_sent_count = [len(sent) for sent in X_sents]  # average sentences in post
        X_word_count = [len(words) for words in X_words]  # average words in post
        X_not_turkish_char = [self.feat_extractor.count_token_contain_notturkish_char(word) for word in
                              X_words]  # in post

        X_word_count_avg = []
        for idx, sents in enumerate(X_sents):
            X_word_count_avg.append(X_word_count[idx] / X_sent_count[idx])

        # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
        X_words_flat = [len(item) for sublist in X_words for item in sublist]

        word_len_mean = np.mean(np.asarray(X_words_flat))

        with open(self.model_dir / self.params_path, 'w', encoding='utf-8') as f:
            json.dump({"word_len_mean": word_len_mean}, f, ensure_ascii=False, indent=4)

        X_long_words = [self.feat_extractor.count_of_long_words(word, word_len_mean) for word in
                        X_words]

        X_short_words = [self.feat_extractor.count_of_short_words(word, word_len_mean) for word in
                         X_words]

        # TODO POS Tags

        assert len(X_sent_count) == len(X_word_count) == len(X_not_turkish_char) == len(X_word_count_avg) == len(
            X_long_words) == len(X_short_words)

        X_sent_count = np.asarray(X_sent_count)
        X_word_count = np.asarray(X_word_count)
        X_not_turkish_char = np.asarray(X_not_turkish_char)
        X_word_count_avg = np.asarray(X_word_count_avg)
        X_long_words = np.asarray(X_long_words)
        X_short_words = np.asarray(X_short_words)

        X_concat = np.vstack((X_sent_count, X_word_count, X_not_turkish_char, X_long_words,
                              X_short_words, X_word_count_avg)).reshape(len(X_train), -1)

        assert len(X_concat) == len(X_train)

        clf = MultinomialNB()
        clf.fit(X_concat, y_train)

        with open(self.model_path, 'wb') as fout:
            pickle.dump(clf, fout)

        logger.info("Model is trained.")

    def predict(self, X_test):
        with open(self.model_dir / self.params_path, 'rb') as param_file:
            params = json.load(param_file)

        word_len_mean = params["word_len_mean"]

        with open(self.model_path, 'rb') as pickle_file:
            model = pickle.load(pickle_file)

        X_sents = [self.feat_extractor.extract_sents(text[0]) for text in X_test]
        X_words = [self.feat_extractor.extract_words(text) for text in X_test]

        X_sent_count = [len(sent) for sent in X_sents]  # average sentences in post
        X_word_count = [len(words) for words in X_words]  # average words in post
        X_not_turkish_char = [self.feat_extractor.count_token_contain_notturkish_char(word) for word in
                              X_words]  # in post

        X_word_count_avg = []
        for idx, sents in enumerate(X_sents):
            X_word_count_avg.append(X_word_count[idx] / X_sent_count[idx])

        X_long_words = [self.feat_extractor.count_of_long_words(word, word_len_mean) for word in
                        X_words]

        X_short_words = [self.feat_extractor.count_of_short_words(word, word_len_mean) for word in
                         X_words]

        # TODO POS Tags

        assert len(X_sent_count) == len(X_word_count) == len(X_not_turkish_char) == len(X_word_count_avg) == len(
            X_long_words) == len(X_short_words)

        X_sent_count = np.asarray(X_sent_count)
        X_word_count = np.asarray(X_word_count)
        X_not_turkish_char = np.asarray(X_not_turkish_char)
        X_word_count_avg = np.asarray(X_word_count_avg)
        X_long_words = np.asarray(X_long_words)
        X_short_words = np.asarray(X_short_words)

        X_concat = np.vstack((X_sent_count, X_word_count, X_not_turkish_char, X_long_words,
                              X_short_words, X_word_count_avg)).reshape(len(X_test), -1)

        preds = model.predict(X_concat)
        return preds


class AllFeatureBased(Model):

    def __init__(self, _config):
        self.feat_extractor = FeatureExtractor()
        self.n_min_char = _config["n_min_char"]
        self.n_max_char = _config["n_max_char"]
        self.n_min_word = _config["n_min_word"]
        self.n_max_word = _config["n_max_word"]
        self.max_features = _config["max_features"]
        self.vectorizer_path = 'all_feat_based_vectorizer_{model_name}.pkl'.format(
            model_name=generate_base_model_name(_config))
        self.params_path = 'all_feat_based_params_{model_name}.pkl'.format(
            model_name=generate_base_model_name(_config))
        self.model_dir = None
        self.model_path = 'all_feat_based_{model_name}.pkl'.format(model_name=generate_base_model_name(_config))

    def fit(self, X_train, y_train):
        char_vectorizer = CountVectorizer(analyzer='char',
                                          ngram_range=(self.n_min_char, self.n_max_char), min_df=1,
                                          max_features=self.max_features)
        word_vectorizer = CountVectorizer(analyzer='word',
                                          ngram_range=(self.n_min_word, self.n_max_word), min_df=1,
                                          max_features=self.max_features)

        ngram_vectorizer = FeatureUnion(
            [("char_vectorizer", char_vectorizer), ("word_vectorizer", word_vectorizer)])

        X_sents = [self.feat_extractor.extract_sents(text[0]) for text in X_train]
        X_words = [self.feat_extractor.extract_words(text) for text in X_train]

        X_sent_count = [len(sent) for sent in X_sents]  # average sentences in post
        X_word_count = [len(words) for words in X_words]  # average words in post
        X_not_turkish_char = [self.feat_extractor.count_token_contain_notturkish_char(word) for word in
                              X_words]  # in post

        X_word_count_avg = []
        for idx, sents in enumerate(X_sents):
            X_word_count_avg.append(X_word_count[idx] / X_sent_count[idx])

        # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
        X_words_flat = [len(item) for sublist in X_words for item in sublist]

        word_len_mean = np.mean(np.asarray(X_words_flat))

        with open(self.model_dir / self.params_path, 'w', encoding='utf-8') as f:
            json.dump({"word_len_mean": word_len_mean}, f, ensure_ascii=False, indent=4)

        X_long_words = [self.feat_extractor.count_of_long_words(word, word_len_mean) for word in
                        X_words]

        X_short_words = [self.feat_extractor.count_of_short_words(word, word_len_mean) for word in
                         X_words]

        # TODO POS Tags

        assert len(X_sent_count) == len(X_word_count) == len(X_not_turkish_char) == len(X_word_count_avg) == len(
            X_long_words) == len(X_short_words)

        X_sent_count = np.asarray(X_sent_count)
        X_word_count = np.asarray(X_word_count)
        X_not_turkish_char = np.asarray(X_not_turkish_char)
        X_word_count_avg = np.asarray(X_word_count_avg)
        X_long_words = np.asarray(X_long_words)
        X_short_words = np.asarray(X_short_words)

        X_ngrams = ngram_vectorizer.fit_transform(X_train.ravel())

        with open(self.model_dir / self.vectorizer_path, 'wb') as fout:
            pickle.dump(ngram_vectorizer, fout)

        logger.info("Vectorizer is trained.")

        clf = MultinomialNB()

        X_concat = np.vstack((X_sent_count, X_word_count, X_not_turkish_char, X_long_words,
                              X_short_words, X_word_count_avg)).reshape(len(X_train), -1)


        X_concat = np.concatenate([X_ngrams.todense(), X_concat], axis=1)


        clf.fit(X_concat, y_train)

        with open(self.model_path, 'wb') as fout:
            pickle.dump(clf, fout)

        logger.info("Model is trained.")

    def predict(self, X_test):
        with open(self.model_dir / self.vectorizer_path, 'rb') as pickle_file:
            ngram_vectorizer = pickle.load(pickle_file)

        X_ngrams = ngram_vectorizer.transform(X_test.ravel())

        with open(self.model_dir / self.params_path, 'rb') as param_file:
            params = json.load(param_file)

        word_len_mean = params["word_len_mean"]

        with open(self.model_path, 'rb') as pickle_file:
            model = pickle.load(pickle_file)

        X_sents = [self.feat_extractor.extract_sents(text[0]) for text in X_test]
        X_words = [self.feat_extractor.extract_words(text) for text in X_test]

        X_sent_count = [len(sent) for sent in X_sents]  # average sentences in post
        X_word_count = [len(words) for words in X_words]  # average words in post
        X_not_turkish_char = [self.feat_extractor.count_token_contain_notturkish_char(word) for word in
                              X_words]  # in post

        X_word_count_avg = []
        for idx, sents in enumerate(X_sents):
            X_word_count_avg.append(X_word_count[idx] / X_sent_count[idx])

        X_long_words = [self.feat_extractor.count_of_long_words(word, word_len_mean) for word in
                        X_words]

        X_short_words = [self.feat_extractor.count_of_short_words(word, word_len_mean) for word in
                         X_words]

        # TODO POS Tags

        assert len(X_sent_count) == len(X_word_count) == len(X_not_turkish_char) == len(X_word_count_avg) == len(
            X_long_words) == len(X_short_words)

        X_sent_count = np.asarray(X_sent_count)
        X_word_count = np.asarray(X_word_count)
        X_not_turkish_char = np.asarray(X_not_turkish_char)
        X_word_count_avg = np.asarray(X_word_count_avg)
        X_long_words = np.asarray(X_long_words)
        X_short_words = np.asarray(X_short_words)

        X_concat = np.vstack((X_sent_count, X_word_count, X_not_turkish_char, X_long_words,
                              X_short_words, X_word_count_avg)).reshape(len(X_test), -1)

        X_concat = np.concatenate([X_ngrams.todense(), X_concat], axis=1)

        preds = model.predict(X_concat)

        return preds


MODELS = {
    "baseline": Baseline,
    "baseline-combined": BaselineCombined,
    "feature-based": LexiconFeatureBased,
    "all": AllFeatureBased
}
