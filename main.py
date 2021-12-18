from sklearn.feature_extraction.text import CountVectorizer
import glob
import string

import re
from srp import SRP
from pca import PCA
from DCT import DCT
from rp import RP
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import stopwords
import seaborn as sns
stop_words = stopwords.stopwords


def preprocess(text):
    # remove header
    out = text.find('Lines:')
    text = text[out + 8:]
    while len(text) > 0 and (text[0] in string.digits or text[0] == " "):
        if len(text) == 0:
            break
        text = text[1:]
    # remove numbers
    text = re.sub(r'[0-9]', '', text)
    # remove emails
    text = re.sub(r"\S*@\S*\s?", "", text)
    # remove punctuation
    text = "".join([c for c in text if c not in string.punctuation])
    # remove names (all words starting with capital letters)
    text = re.sub(r"(\b[A-Z][a-z]+('s)?\b)", "", text)
    # lowercase
    text = "".join([c.lower() for c in text])
    # remove words based on length
    text = " ".join([w for w in text.split() if len(w) > 2 and len(w) < 12])
    # remove stopwords
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

def ft(corpus):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus).T
    # remove words with very low frequency
    X = X[np.array(X.sum(axis=1) > 1).squeeze(), :]
    # take 5000 most frequent words
    X = X[np.array(np.argsort(X.sum(axis=1).squeeze())).squeeze()[-5000:], :]
    # remove empty docs
    X = X[:, np.array(X.sum(axis=0) > 0).squeeze()]
    # normalize
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
    pairs = 100
    X = ft(corpus)
    dim = X.shape[0]
    start = 100
    stop = 1000
    step = 20
    y_dct = np.zeros((int((stop - start)/step), pairs))
    y_pca = np.zeros((int((stop - start)/step), pairs))
    y_srp = np.zeros((int((stop - start)/step), pairs))
    y_rp = np.zeros((int((stop - start)/step), pairs))
    for i, k in enumerate(tqdm(range(start, stop, step))):
        """srp = SRP(X, k)
        srp.fit()
        y_srp[k] = srp.distortions_inner_product(pairs)"""
        pca = PCA(X, k)
        pca.fit()
        y_pca[i] = pca.distortions_inner_product(pairs)
        """dct = DCT(X, k)
        dct.fit()
        y_dct[k] = dct.distortions_inner_product(pairs)"""
        rp = RP(X, k)
        rp.fit()
        y_rp[i] = rp.distortions_inner_product(pairs)


    x = np.arange(100, 1000, 20)
    ci_rp = 1.96 * np.std(y_rp, axis=1) / np.sqrt(pairs)
    ci_pca = 1.96 * np.std(y_pca, axis=1) / np.sqrt(pairs)
    plt.scatter(x, np.mean(y_rp, axis=1), marker="+")
    plt.scatter(x, np.mean(y_pca, axis=1), marker="d")
    plt.fill_between(x, (np.mean(y_rp, axis=1) - ci_rp), (np.mean(y_rp, axis=1) + ci_rp), color='blue', alpha=0.1)
    plt.fill_between(x, (np.mean(y_pca, axis=1) - ci_pca), (np.mean(y_pca, axis=1) + ci_pca), color='orange', alpha=0.1)
    plt.legend(["RP", "SVD"])
    plt.title("Average error using RP and SVD")
    plt.xlabel("Reduced dim. of data")
    plt.ylabel("Error difference")
    plt.show()