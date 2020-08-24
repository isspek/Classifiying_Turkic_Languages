import string

from pyzemberek.zemberek.stemmer import Stemmer
from pyzemberek.zemberek.tokenization import Tokenization

TURKISH_CHARS = ['a', 'b', 'c', 'ç', 'd', 'e', 'f', 'g', 'ğ', 'h', 'ı', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'ö'
                                                                                                           'p', 'r',
                 's', 'ş', 't', 'u', 'ü', 'v', 'y', 'z']


class FeatureExtractor:
    def __init__(self):
        self.tokenizer = Tokenization()
        self.stemmer = Stemmer()

    def _remove_punctuations(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def extract_words(self, text):
        text = str(text)
        text = self._remove_punctuations(text)
        word_tokens = self.tokenizer.word_tokenize(text)
        return word_tokens

    def extract_sents(self, text):
        text = str(text)
        sents = self.tokenizer.sentence_tokenize(text)
        return sents

    def count_token_contain_notturkish_char(self, tokens):
        count = 0
        for token in tokens:
            for _char in token:
                if _char not in TURKISH_CHARS:
                    count += 1
        return count

    def count_of_long_words(self, tokens, th):
        count = 0
        for token in tokens:
            if len(token) > th:
                count += 1
        return count

    def count_of_short_words(self, tokens, th):
        count = 0
        for token in tokens:
            if len(token) < th:
                count += 1
        return count
