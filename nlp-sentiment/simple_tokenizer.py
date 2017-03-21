import nltk
import numpy as np

class SimpleTokenizer():
    def tokens_to_vector(self, tokens, label, word_index_map):
        xy = np.zeros(len(word_index_map) + 1)
        for t in tokens:
            i = word_index_map[t]
            xy[i] += 1
        xy = xy / xy.sum()
        xy[-1] = label
        return xy

    def tokenize(self, s, wordnet_lemmatizer, stopwords):
        s = s.lower()
        tokens = nltk.tokenize.word_tokenize(s)
        tokens = [t for t in tokens if len(t) > 2]
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if t not in stopwords]
        return tokens
