from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import *
from data_sanitizer import SimpleDataModule
from simple_tokenizer import SimpleTokenizer
import numpy as np
import nltk

def analyze():
    wordnet_lemmatizer = WordNetLemmatizer()
    stopwords = set(w.rstrip() for w in open('stopwords.txt'))

    tokenizer = SimpleTokenizer()

    csv = SimpleDataModule('reviews.csv')
    cols = csv.load_data()

    positive_reviews = cols[0]
    negative_reviews = cols[1]

    word_index_map = {}
    current_index = 0

    positive_tokens = []
    negative_tokens = []

    for review in positive_reviews:
        tokens = tokenizer.tokenize(review, wordnet_lemmatizer, stopwords)
        positive_tokens.append(tokens)
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1

    for review in negative_reviews:
        tokens = tokenizer.tokenize(review, wordnet_lemmatizer, stopwords)
        negative_tokens.append(tokens)
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1


    N = len(positive_tokens) + len(negative_tokens)
    data = np.zeros((N, len(word_index_map) + 1))

    i = 0
    for tokens in positive_tokens:
        xy = tokenizer.tokens_to_vector(tokens, 1, word_index_map)
        data[i,:] = xy
        i += 1

    for tokens in negative_tokens:
        xy = tokenizer.tokens_to_vector(tokens, 0, word_index_map)
        data[i,:] = xy
        i += 1

    np.random.shuffle(data)

    X = data[:, :-1]
    Y = data[:, -1]

    Xtrain = X[:-100,]
    Ytrain = Y[:-100,]
    Xtest = X[-100:,]
    Ytest = Y[-100:,]

    model = LogisticRegression()
    model.fit(Xtrain, Ytrain)
    print "Classification rate: ", model.score(Xtest, Ytest)

    threshold = 0.5
    
    for word, index in word_index_map.iteritems():
        weight = model.coef_[0][index]
        if weight > threshold or weight < -threshold:
            print word, weight