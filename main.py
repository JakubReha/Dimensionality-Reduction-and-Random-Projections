from sklearn.feature_extraction.text import CountVectorizer
import glob
import string
from stopwords import get_stopwords
import re
from srp import SRP
from pca import PCA
from DCT import DCT
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize

stop_words = get_stopwords('en')


def preprocess(text):
    # remove header
    out = text.find('Lines:')
    text = text[out + 8:]
    while len(text) > 0 and (text[0] in string.digits or text[0] == " "):
        if len(text) == 0:
            break
        text = text[1:]
    # remove emails
    text = re.sub(r"\S*@\S*\s?", "", text)
    # remove punctuation
    text = "".join([c for c in text if c not in string.punctuation])
    # lowercase
    text = "".join([c.lower() for c in text])
    # remove stopwords
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

def ft(corpus):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus).T
    X = X[:, np.array(X.sum(axis=0) > 0).squeeze()]
    X = np.array(X / np.sum(X, axis=0))
    return X

if __name__ == "__main__":
    docs = glob.glob('data/text/**/*')
    corpus = []
    for doc in docs:
        try:
            with open(doc) as f:
                corpus.append(preprocess(" ".join([l.rstrip() for l in f])))
        except:
            pass
    y = []
    X = ft(corpus)
    dim = X.shape[0]
    for k in tqdm(range(100, 1000, 20)):
        """srp = SRP(dim, k)
        srp.project(X)
        y.append(np.mean(srp.distortions_inner_product(90)))
        srp = PCA(X, k)
        srp.fit()
        y.append(np.mean(srp.distortions_inner_product(90)))"""
        srp = PCA(X, k)
        srp.fit()
        y.append(np.mean(srp.distortions_inner_product(90)))

    x = np.arange(100, 1000, 20)
    plt.scatter(x, y)
    plt.show()