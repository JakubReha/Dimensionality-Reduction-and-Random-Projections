from sklearn.feature_extraction.text import CountVectorizer
import glob
import string
from PIL import Image
import re
from srp import SRP
from PCA import PCA
from DCT import DCT
from rp import RP
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import stopwords
import time
import sys
import seaborn as sns
stop_words = stopwords.stopwords
# from python_papi import events, papi_high as high



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

def test_text():
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
    start = 1
    stop = 801
    step = 20
    y_dct = np.zeros((int((stop - start) / step), pairs))
    y_pca = np.zeros((int((stop - start) / step), pairs))
    y_srp = np.zeros((int((stop - start) / step), pairs))
    y_rp = np.zeros((int((stop - start) / step), pairs))
    for i, k in enumerate(tqdm(range(start, stop, step))):
        """srp = SRP(X, k)
        srp.fit()
        y_srp[i] = srp.distortions_inner_product(pairs)"""
        pca = PCA(X, k)
        pca.fit()
        y_pca[i] = pca.distortions_inner_product(pairs)
        """dct = DCT(X, k)
        dct.fit()
        y_dct[i] = dct.distortions_inner_product(pairs)"""
        rp = RP(X, k)
        rp.fit()
        y_rp[i] = rp.distortions_inner_product(pairs)

    x = np.arange(start, stop, step)
    ci_rp = 1.96 * np.std(y_rp, axis=1) / np.sqrt(pairs)
    ci_pca = 1.96 * np.std(y_pca, axis=1) / np.sqrt(pairs)
    # ci_srp = 1.96 * np.std(y_srp, axis=1) / np.sqrt(pairs)
    plt.scatter(x, np.mean(y_rp, axis=1), marker="+")
    plt.scatter(x, np.mean(y_pca, axis=1), marker="d")
    # plt.scatter(x, np.mean(y_srp, axis=1), marker="o")
    plt.fill_between(x, (np.mean(y_rp, axis=1) - ci_rp), (np.mean(y_rp, axis=1) + ci_rp), color='blue', alpha=0.1)
    plt.fill_between(x, (np.mean(y_pca, axis=1) - ci_pca), (np.mean(y_pca, axis=1) + ci_pca), color='orange', alpha=0.1)
    # plt.fill_between(x, (np.mean(y_srp, axis=1) - ci_srp), (np.mean(y_srp, axis=1) + ci_srp), color='red', alpha=0.1)
    plt.legend(["RP", "SVD"])
    plt.title("Average error using RP and SVD")
    plt.xlabel("Reduced dim. of data")
    plt.ylabel("Error difference")
    plt.show()

def test_image():
    data_images = []
    imgs = glob.glob('data/img/*.tiff')
    for img in imgs:
        im = Image.open(img)
        image_arr = np.array(im)
        #TODO: normalize or not?
        flat_arr = image_arr.flatten()/255
        data_images.append(flat_arr)
    X = np.array(data_images).T
    pairs = 100
    start = 1
    stop = 801
    step = 20
    y_dct = np.zeros((int((stop - start) / step), pairs))
    y_pca = np.zeros((int((stop - start) / step), pairs))
    y_srp = np.zeros((int((stop - start) / step), pairs))
    y_rp = np.zeros((int((stop - start) / step), pairs))
    flops = np.zeros((4,40))

    for i, k in enumerate(tqdm(range(start, stop, step))):
        starttime = time.time()
        srp = SRP(X, k)
        srp.fit()
        flops[0,i] = time.time() - starttime
        y_srp[i] = srp.distortions_euclidean(pairs)
        starttime = time.time()
        pca = PCA(X, k)
        pca.fit()
        flops[1,i] = time.time() - starttime
        y_pca[i] = pca.distortions_euclidean(pairs)
        starttime = time.time()
        dct = DCT(X, k)
        dct.fit()
        flops[2,i] = time.time() - starttime
        y_dct[i] = dct.distortions_euclidean(pairs)

        starttime = time.time()
        rp = RP(X, k)
        rp.fit()
        flops[3,i] = time.time() - starttime
        y_rp[i] = rp.distortions_euclidean(pairs)


    x = np.arange(start, stop, step)
    ci_rp = 1.96 * np.std(y_rp, axis=1) / np.sqrt(pairs)
    ci_pca = 1.96 * np.std(y_pca, axis=1) / np.sqrt(pairs)
    ci_srp = 1.96 * np.std(y_srp, axis=1) / np.sqrt(pairs)
    ci_dct = 1.96 * np.std(y_dct, axis=1) / np.sqrt(pairs)


    plt.scatter(x, np.mean(y_rp, axis=1), marker="+")
    plt.scatter(x, np.mean(y_pca, axis=1), marker="d")
    plt.scatter(x, np.mean(y_srp, axis=1), marker="o")
    plt.scatter(x, np.mean(y_dct, axis=1), marker=".")

    plt.fill_between(x, (np.mean(y_rp, axis=1) - ci_rp), (np.mean(y_rp, axis=1) + ci_rp), color='blue', alpha=0.1)
    plt.fill_between(x, (np.mean(y_pca, axis=1) - ci_pca), (np.mean(y_pca, axis=1) + ci_pca), color='orange', alpha=0.1)
    plt.fill_between(x, (np.mean(y_srp, axis=1) - ci_srp), (np.mean(y_srp, axis=1) + ci_srp), color='red', alpha=0.1)
    plt.fill_between(x, (np.mean(y_dct, axis=1) - ci_dct), (np.mean(y_dct, axis=1) + ci_dct), color='yellow', alpha=0.1)

    plt.legend(["RP", "PCA", "SRP", "DCT"])
    # plt.legend(["PCA", "DCT"])
    plt.title("Average error using PCA, RP, SVD and DCT")
    plt.xlabel("Reduced dim. of data")
    plt.ylabel("Error difference")
    plt.show()

    plt.scatter(x, flops[0, :], marker="o")
    plt.scatter(x, flops[1, :], marker="d")
    plt.scatter(x, flops[2, :], marker=".")
    plt.scatter(x, flops[3, :], marker="+")
    plt.legend(["SRP", "PCA", "DCT", "RP"])
    plt.title("Running time using PCA, RP, SVD and DCT")
    plt.xlabel("Reduced dim. of data")
    plt.ylabel("running time (seconds)")
    plt.show()

if __name__ == "__main__":
    test_image()
    #test_text()