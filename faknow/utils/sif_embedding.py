import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD

ASSETS_PATH = Path(__file__).parent.parent.parent / "assets"
Words = None
We = None
Weight4ind = None


class Params(object):
    def __init__(self):
        """Class to hold parameter values for SIF embedding."""
        self.LW = 1e-5
        self.LC = 1e-5
        self.eta = 0.05

    def __str__(self):
        t = "LW", self.LW, ", LC", self.LC, ", eta", self.eta
        t = map(str, t)
        return ' '.join(t)


params = Params()
params.rmpc = 1


def getWordWeight(weightfile, a=1e-3):
    """
    Get word weights from a file and apply a weight transformation.

    Args:
        weightfile (str): Path to the file containing word weights.
        a (float, optional): Smoothing parameter for word weights (default: 1e-3).

    Returns:
        dict: A dictionary mapping words to their corresponding weights.
    """ 
    if a <= 0:  # when the parameter makes no sense, use unweighted
        a = 1.0

    word2weight = {}
    with open(weightfile) as f:
        lines = f.readlines()
    N = 0
    for i in lines:
        i = i.strip()
        if (len(i) > 0):
            i = i.split()
            if (len(i) == 2):
                word2weight[i[0]] = float(i[1])
                N += float(i[1])
            else:
                print(i)
    for key, value in word2weight.items():
        word2weight[key] = a / (a + value / N)
    return word2weight


def getWeight(words, word2weight):
    """
    Get word weights for words in the given dictionary.

    Args:
        words (dict): A dictionary mapping words to their indices.
        word2weight (dict): A dictionary mapping words to their corresponding weights.

    Returns:
        dict: A dictionary mapping word indices to their corresponding weights.
    """
    weight4ind = {}
    for word, ind in words.items():
        if word in word2weight:
            weight4ind[ind] = word2weight[word]
        else:
            weight4ind[ind] = 1.0
    return weight4ind


def lookupIDX(words, w):
    """
    Look up the index of a word in the given dictionary.

    Args:
        words (dict): A dictionary mapping words to their indices.
        w (str): The word to look up.

    Returns:
        int: The index of the word if found, otherwise the index for 'UUUNKKK' (unknown word).
    """
    w = w.lower()
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#", "")
    if w in words:
        return words[w]
    elif 'UUUNKKK' in words:
        return words['UUUNKKK']
    else:
        return len(words) - 1


def getSeq(p1, words):
    """
    Get the sequence of word indices for a given sentence.

    Args:
        p1 (list): A list of words in the sentence.
        words (dict): A dictionary mapping words to their indices.

    Returns:
        list: A list of word indices in the sentence.
    """
    # p1 = p1.split()
    X1 = []
    for i in p1:
        X1.append(lookupIDX(words, i))
    return X1


def prepare_data(list_of_seqs):
    """
    Prepare input data for the model.

    Args:
        list_of_seqs (list): A list of sequences containing word indices.

    Returns:
        tuple: A tuple containing the input data (x) and the mask (x_mask).
    """
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    maxlen = np.max(lengths)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.
    x_mask = np.asarray(x_mask, dtype='float32')
    return x, x_mask


def sentences2idx(sentences, words):
    """
    Convert a list of sentences to a numpy array of word indices.

    Args:
        sentences (list): A list of sentences, each represented as a list of words.
        words (dict): A dictionary mapping words to their indices.

    Returns:
        tuple: A tuple containing the input data (x) and the mask (m).
    """
    seq1 = []
    for i in sentences:
        seq1.append(getSeq(i, words))
    x1, m1 = prepare_data(seq1)
    return x1, m1


def seq2weight(seq, mask, weight4ind):
    """
    Convert a sequence and its mask to a weight matrix.

    Args:
        seq (numpy.ndarray): A numpy array containing word indices for a sequence.
        mask (numpy.ndarray): A numpy array representing the mask for the sequence.
        weight4ind (dict): A dictionary mapping word indices to their corresponding weights.

    Returns:
        numpy.ndarray: A weight matrix for the sequence.
    """
    weight = np.zeros(seq.shape).astype('float32')
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            if mask[i, j] > 0 and seq[i, j] >= 0:
                weight[i, j] = weight4ind[seq[i, j]]
    weight = np.asarray(weight, dtype='float32')
    return weight


def get_weighted_average(We, x, w):
    """
    Compute the weighted average vectors.

    Args:
        We (numpy.ndarray): A numpy array containing word embeddings.
        x (numpy.ndarray): A numpy array of word indices for sentences.
        w (numpy.ndarray): A weight matrix for the words in sentences.

    Returns:
        numpy.ndarray: Weighted average vectors for the sentences.
    """
    n_samples = x.shape[0]
    emb = np.zeros((n_samples, We.shape[1]))
    for i in range(n_samples):
        emb[i, :] = w[i, :].dot(We[x[i, :], :]) / np.count_nonzero(w[i, :])
    return emb


def compute_pc(X, npc=1):
    """
    Compute the principal components.

    Args:
        X (numpy.ndarray): A numpy array containing data points.
        npc (int, optional): Number of principal components to compute (default: 1).

    Returns:
        numpy.ndarray: Array containing principal components.
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_


def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components.

    Args:
        X (numpy.ndarray): A numpy array containing data points.
        npc (int, optional): Number of principal components to remove (default: 1).

    Returns:
        numpy.ndarray: Array containing data points after removing their projection on the principal components.
    """
    pc = compute_pc(X, npc)
    if npc == 1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


def SIF_embedding(We, x, w, params):
    """
    Compute the SIF (Smooth Inverse Frequency) embeddings for sentences.

    Args:
        We (numpy.ndarray): A numpy array containing word embeddings.
        x (numpy.ndarray): A numpy array of word indices for sentences.
        w (numpy.ndarray): A weight matrix for the words in sentences.
        params (Params): An instance of the Params class containing the parameters.

    Returns:
        numpy.ndarray: SIF embeddings for sentences.
    """
    emb = get_weighted_average(We, x, w)
    if params.rmpc > 0:
        emb = remove_pc(emb, params.rmpc)
    return emb


def sif_embedding(sentences):
    """
    Compute SIF embeddings for a list of sentences.

    Args:
        sentences (list): A list of sentences, each represented as a list of words.

    Returns:
        numpy.ndarray: SIF embeddings for the given sentences.
    """
    global Words, We, Weight4ind

    if Words is None:
        with open(ASSETS_PATH / "words.json") as f:
            Words = json.load(f)

    if We is None:
        We = np.load(ASSETS_PATH / "We.npy")

    if Weight4ind is None:
        with open(ASSETS_PATH / "weight4ind.pkl", "rb") as f:
            Weight4ind = pickle.load(f)

    x, m = sentences2idx(sentences, Words)
    w = seq2weight(x, m, Weight4ind)
    return SIF_embedding(We, x, w, params)


def preprocess():
    """
    Preprocess the data to calculate word weights.

    This function is run once to calculate and save word weights to be used later during SIF embedding.
    """
    with open(ASSETS_PATH / "words.json") as f:
        Words = json.load(f)

    word2weight = getWordWeight(
        ASSETS_PATH / "enwiki_vocab_min200.txt",
        1e-3,  # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
    )
    weight4ind = getWeight(Words, word2weight)

    with open(ASSETS_PATH / "weight4ind.pkl", "wb") as f:
        pickle.dump(weight4ind, f)


def main():
    """
    Main function to demonstrate SIF embedding on a sample sentence.
    """
    embd = sif_embedding(
        ["the parameter in the SIF weighting scheme, usually in the range"])
    print(embd.shape)


if __name__ == "__main__":
    # preprocess()  # Uncomment this line to run the preprocess step (only needs to be done once)
    main()
