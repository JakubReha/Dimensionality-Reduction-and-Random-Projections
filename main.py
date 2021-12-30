from sklearn.feature_extraction.text import CountVectorizer
import glob
import string
from PIL import Image
import re
import pandas as pd
from srp import SRP
from PCA import PCA
from DCT import DCT
from rp import RP
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import stopwords
from scipy.io import wavfile
from scipy.ndimage import median_filter
import seaborn as sns
stop_words = stopwords.stopwords
np.random.seed(2)


def preprocess(text, header=False):
    # remove header
    if header:
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
                corpus.append(preprocess(" ".join([l.rstrip() for l in f]), True))
        except:
            pass
    pairs = 100
    X = ft(corpus)
    dim = X.shape[0]
    start = 1
    stop = 801
    step = 40
    y_dct = np.zeros((int((stop - start) / step), pairs))
    y_pca = np.zeros((int((stop - start) / step), pairs))
    y_srp = np.zeros((int((stop - start) / step), pairs))
    y_rp = np.zeros((int((stop - start) / step), pairs))
    #dct = DCT(X)
    pca = PCA(X, stop)
    for i, k in enumerate(tqdm(range(start, stop, step))):
        """srp = SRP(X, k)
        srp.fit()
        y_srp[i] = srp.distortions_inner_product(pairs)"""
        pca.fit(k)
        y_pca[i] = pca.distortions_inner_product(pairs)
        """dct.fit(k)
        y_dct[i] = dct.distortions_inner_product(pairs)"""
        rp = RP(X, k)
        rp.fit()
        y_rp[i] = rp.distortions_inner_product(pairs)

    x = np.arange(start, stop, step)
    ci_rp = 1.96 * np.std(y_rp, axis=1) / np.sqrt(pairs)
    ci_pca = 1.96 * np.std(y_pca, axis=1) / np.sqrt(pairs)
    #ci_dct = 1.96 * np.std(y_dct, axis=1) / np.sqrt(pairs)
    #ci_srp = 1.96 * np.std(y_srp, axis=1) / np.sqrt(pairs)
    #plt.scatter(x, np.mean(y_dct, axis=1), marker="d")
    plt.scatter(x, np.mean(y_rp, axis=1), marker="+")
    plt.scatter(x, np.mean(y_pca, axis=1), marker="d")
    #plt.scatter(x, np.mean(y_srp, axis=1), marker="o")
    #plt.fill_between(x, (np.mean(y_dct, axis=1) - ci_dct), (np.mean(y_dct, axis=1) + ci_dct), color='blue', alpha=0.1)
    plt.fill_between(x, (np.mean(y_rp, axis=1) - ci_rp), (np.mean(y_rp, axis=1) + ci_rp), color='blue', alpha=0.1)
    plt.fill_between(x, (np.mean(y_pca, axis=1) - ci_pca), (np.mean(y_pca, axis=1) + ci_pca), color='orange', alpha=0.1)
    #plt.fill_between(x, (np.mean(y_srp, axis=1) - ci_srp), (np.mean(y_srp, axis=1) + ci_srp), color='red', alpha=0.1)
    plt.legend(["RP", "PCA"])
    plt.title("Average error using RP and SVD")
    plt.xlabel("Reduced dim. of data")
    plt.ylabel("Error difference")
    plt.show()

def test_image(type="img"):
    data_images = []
    mf_images = []
    imgs = glob.glob('data/{}/*.tiff'.format(type))
    for img in imgs:
        im = Image.open(img)
        image_arr = np.array(im)/255
        flat_arr = image_arr.flatten()
        data_images.append(flat_arr)
        mf_images.append(median_filter(image_arr, (3, 3)).flatten())
    X = np.array(data_images).T
    pairs = 100
    start = 1
    stop = 801
    step = 20
    y_dct = np.zeros((int((stop - start) / step), pairs))
    y_pca = np.zeros((int((stop - start) / step), pairs))
    y_srp = np.zeros((int((stop - start) / step), pairs))
    y_rp = np.zeros((int((stop - start) / step), pairs))
    x = np.arange(start, stop, step)
    if type == "noisy_img":
        X_mf = np.array(mf_images).T
        ids1 = np.random.choice(range(X_mf.shape[1]), size=pairs)
        ids2 = np.random.choice(list(set(range(X_mf.shape[1])) - set(ids1)), size=pairs)
        emd_dist = np.sqrt(np.sum((X_mf.T[ids1] - X_mf.T[ids2]) ** 2, axis=1))
        dist = np.sqrt(np.sum((X.T[ids1] - X.T[ids2]) ** 2, axis=1))
        y_mf = emd_dist - dist
        ci_mf = 1.96 * np.std(y_mf) / np.sqrt(pairs)
        plt.plot(x, np.ones_like(x)*np.mean(y_mf), color="black")

    dct = DCT(X)
    pca = PCA(X, stop)
    for i, k in enumerate(tqdm(range(start, stop, step))):
        srp = SRP(X, k)
        srp.fit()
        y_srp[i] = srp.distortions_euclidean(pairs)
        pca.fit(k)
        y_pca[i] = pca.distortions_euclidean(pairs)
        dct.fit(k)
        y_dct[i] = dct.distortions_euclidean(pairs)
        rp = RP(X, k)
        rp.fit()
        y_rp[i] = rp.distortions_euclidean(pairs)

    ci_rp = 1.96 * np.std(y_rp, axis=1) / np.sqrt(pairs)
    ci_pca = 1.96 * np.std(y_pca, axis=1) / np.sqrt(pairs)
    ci_srp = 1.96 * np.std(y_srp, axis=1) / np.sqrt(pairs)
    ci_dct = 1.96 * np.std(y_dct, axis=1) / np.sqrt(pairs)

    plt.scatter(x, np.mean(y_rp, axis=1), marker="+", color="blue")
    plt.scatter(x, np.mean(y_pca, axis=1), marker="d", color="orange")
    plt.scatter(x, np.mean(y_srp, axis=1), marker="o", color="red")
    plt.scatter(x, np.mean(y_dct, axis=1), marker=".", color="green")

    plt.fill_between(x, (np.mean(y_rp, axis=1) - ci_rp), (np.mean(y_rp, axis=1) + ci_rp), color='blue', alpha=0.1)
    plt.fill_between(x, (np.mean(y_pca, axis=1) - ci_pca), (np.mean(y_pca, axis=1) + ci_pca), color='orange', alpha=0.1)
    plt.fill_between(x, (np.mean(y_srp, axis=1) - ci_srp), (np.mean(y_srp, axis=1) + ci_srp), color='red', alpha=0.1)
    plt.fill_between(x, (np.mean(y_dct, axis=1) - ci_dct), (np.mean(y_dct, axis=1) + ci_dct), color='green', alpha=0.1)

    if type == "noisy_img":
        plt.fill_between(x, (np.mean(y_mf) - ci_mf), (np.mean(y_mf) + ci_mf), color='black', alpha=0.1)
        plt.legend(["MF", "RP", "PCA", "SRP", "DCT"])
        plt.title("Average error using MF, PCA, RP, SVD and DCT")
    else:
        plt.legend(["RP", "PCA", "SRP", "DCT"])
        plt.title("Average error using PCA, RP, SVD and DCT")
    plt.xlabel("Reduced dim. of data")
    plt.ylabel("Error difference")
    plt.show()

def test_audio():
    files = glob.glob('data/audio/**/*.wav')
    dim = 100000
    X = np.zeros((dim, len(files)))
    for i, file in enumerate(files):
        rate, data = wavfile.read(file)
        try:
            X[:, i] = data[:dim]
        except:
            pass
    X = X[:, X.sum(axis=0) != 0]
    X = X/(2**16-1)
    pairs = 100
    start = 1
    stop = 20001
    step = 1000
    y_dct = np.zeros((int((stop - start) / step), pairs))
    dct = DCT(X)
    start2 = 1
    #stop2 = 2601
    #step2 = 200
    stop2 = 1001
    step2 = 50
    pca = PCA(X, stop2)
    y_pca = np.zeros((int((stop2 - start2) / step2), pairs))
    y_srp = np.zeros((int((stop2 - start2) / step2), pairs))
    y_rp = np.zeros((int((stop2 - start2) / step2), pairs))
    for i, k in enumerate(tqdm(range(start2, stop2, step2))):
        srp = SRP(X, k)
        srp.fit()
        y_srp[i] = srp.distortions_euclidean(pairs)
        pca.fit(k)
        y_pca[i] = pca.distortions_euclidean(pairs)
        rp = RP(X, k)
        rp.fit()
        y_rp[i] = rp.distortions_euclidean(pairs)
    #for i, k in enumerate(tqdm(range(start, stop, step))):
        dct.fit(k)
        y_dct[i] = dct.distortions_euclidean(pairs)

    #x_dct = np.arange(start, stop, step)
    x = np.arange(start2, stop2, step2)
    #x_add = np.arange(stop2, stop, step)
    #y_add = np.ones_like(x_add)
    ci_rp = 1.96 * np.std(y_rp, axis=1) / np.sqrt(pairs)
    ci_pca = 1.96 * np.std(y_pca, axis=1) / np.sqrt(pairs)
    ci_srp = 1.96 * np.std(y_srp, axis=1) / np.sqrt(pairs)
    ci_dct = 1.96 * np.std(y_dct, axis=1) / np.sqrt(pairs)

    """plt.scatter(np.concatenate((x, x_add)), np.concatenate((np.mean(y_rp, axis=1), y_add * np.mean(y_rp, axis=1)[-1])), marker="+", color="blue", alpha=0.7)
    plt.scatter(np.concatenate((x, x_add)), np.concatenate((np.mean(y_pca, axis=1), y_add * np.mean(y_pca, axis=1)[-1])), marker="d", color="orange", alpha=0.7)
    plt.scatter(np.concatenate((x, x_add)), np.concatenate((np.mean(y_srp, axis=1), y_add * np.mean(y_srp, axis=1)[-1])), marker="o", color="red", alpha=0.7)
    plt.scatter(x_dct, np.mean(y_dct, axis=1), marker=".", color="green", alpha=0.7)

    plt.fill_between(np.concatenate((x, x_add)), np.concatenate(((np.mean(y_rp, axis=1) - ci_rp), y_add*(np.mean(y_rp, axis=1) - ci_rp)[-1])), np.concatenate(((np.mean(y_rp, axis=1) + ci_rp), y_add*(np.mean(y_rp, axis=1) + ci_rp)[-1])), color='blue', alpha=0.1)
    plt.fill_between(np.concatenate((x, x_add)), np.concatenate(((np.mean(y_pca, axis=1) - ci_pca), y_add*(np.mean(y_pca, axis=1) - ci_pca)[-1])), np.concatenate(((np.mean(y_pca, axis=1) + ci_pca), y_add*(np.mean(y_pca, axis=1) + ci_pca)[-1])), color='orange', alpha=0.1)
    plt.fill_between(np.concatenate((x, x_add)), np.concatenate(((np.mean(y_srp, axis=1) - ci_srp), y_add*(np.mean(y_srp, axis=1) - ci_srp)[-1])), np.concatenate(((np.mean(y_srp, axis=1) + ci_srp), y_add*(np.mean(y_srp, axis=1) + ci_srp)[-1])), color='red', alpha=0.1)
    plt.fill_between(x_dct, (np.mean(y_dct, axis=1) - ci_dct), (np.mean(y_dct, axis=1) + ci_dct), color='green', alpha=0.1)"""

    plt.scatter(x, np.mean(y_rp, axis=1), marker="+", color="blue")
    plt.scatter(x, np.mean(y_pca, axis=1), marker="d", color="orange")
    plt.scatter(x, np.mean(y_srp, axis=1), marker="o", color="red")
    plt.scatter(x, np.mean(y_dct, axis=1), marker=".", color="green")

    plt.fill_between(x, (np.mean(y_rp, axis=1) - ci_rp), (np.mean(y_rp, axis=1) + ci_rp), color='blue', alpha=0.1)
    plt.fill_between(x, (np.mean(y_pca, axis=1) - ci_pca), (np.mean(y_pca, axis=1) + ci_pca), color='orange', alpha=0.1)
    plt.fill_between(x, (np.mean(y_srp, axis=1) - ci_srp), (np.mean(y_srp, axis=1) + ci_srp), color='red', alpha=0.1)
    plt.fill_between(x, (np.mean(y_dct, axis=1) - ci_dct), (np.mean(y_dct, axis=1) + ci_dct), color='green', alpha=0.1)

    plt.legend(["RP", "PCA", "SRP", "DCT"])
    plt.title("Average error using PCA, RP, SVD and DCT")
    plt.xlabel("Reduced dim. of data")
    plt.ylabel("Error difference")
    plt.show()


def text2():
    docs = glob.glob('data/text2/**/*')
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
    step = 40
    y_dct = np.zeros((int((stop - start) / step), pairs))
    y_pca = np.zeros((int((stop - start) / step), pairs))
    y_srp = np.zeros((int((stop - start) / step), pairs))
    y_rp = np.zeros((int((stop - start) / step), pairs))
    # dct = DCT(X)
    pca = PCA(X, stop)
    for i, k in enumerate(tqdm(range(start, stop, step))):
        """srp = SRP(X, k)
        srp.fit()
        y_srp[i] = srp.distortions_inner_product(pairs)"""
        pca.fit(k)
        y_pca[i] = pca.distortions_inner_product(pairs)
        """dct.fit(k)
        y_dct[i] = dct.distortions_inner_product(pairs)"""
        rp = RP(X, k)
        rp.fit()
        y_rp[i] = rp.distortions_inner_product(pairs)

    x = np.arange(start, stop, step)
    ci_rp = 1.96 * np.std(y_rp, axis=1) / np.sqrt(pairs)
    ci_pca = 1.96 * np.std(y_pca, axis=1) / np.sqrt(pairs)
    # ci_dct = 1.96 * np.std(y_dct, axis=1) / np.sqrt(pairs)
    # ci_srp = 1.96 * np.std(y_srp, axis=1) / np.sqrt(pairs)
    # plt.scatter(x, np.mean(y_dct, axis=1), marker="d")
    plt.scatter(x, np.mean(y_rp, axis=1), marker="+")
    plt.scatter(x, np.mean(y_pca, axis=1), marker="d")
    # plt.scatter(x, np.mean(y_srp, axis=1), marker="o")
    # plt.fill_between(x, (np.mean(y_dct, axis=1) - ci_dct), (np.mean(y_dct, axis=1) + ci_dct), color='blue', alpha=0.1)
    plt.fill_between(x, (np.mean(y_rp, axis=1) - ci_rp), (np.mean(y_rp, axis=1) + ci_rp), color='blue', alpha=0.1)
    plt.fill_between(x, (np.mean(y_pca, axis=1) - ci_pca), (np.mean(y_pca, axis=1) + ci_pca), color='orange', alpha=0.1)
    # plt.fill_between(x, (np.mean(y_srp, axis=1) - ci_srp), (np.mean(y_srp, axis=1) + ci_srp), color='red', alpha=0.1)
    plt.legend(["RP", "PCA"])
    plt.title("Average error using RP and SVD")
    plt.xlabel("Reduced dim. of data")
    plt.ylabel("Error difference")
    plt.show()


if __name__ == "__main__":
    #text2()
    test_image("img")
    #test_audio()
    #test_image("noisy_img")
    #test_text()