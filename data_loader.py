import scipy.io as scio
import numpy as np

UMIST = 'UMIST'
COIL20 = 'COIL20'
JAFFE = 'JAFFE'
PALM = 'Palm'
USPS = 'USPSdata_20_uni'
MNIST_TEST = 'mnist_test'
SEGMENT = 'segment_uni'
NEWS = '20news_uni'
TEXT = 'text1_uni'
ISOLET = 'Isolet'
BALSAC = 'balsac'


def load_cora():
    path = 'data/cora.mat'
    data = scio.loadmat(path)
    labels = data['gnd']
    labels = np.reshape(labels, (labels.shape[0],))
    X = data['fea']
    X = X.astype(np.float32)
    X /= np.max(X)
    links = data['W']
    return X, labels, links


def load_data(name):
    path = 'data/{}.mat'.format(name)
    data = scio.loadmat(path)
    labels = data['Y']
    labels = np.reshape(labels, (labels.shape[0],))
    X = data['X']
    X = X.astype(np.float32)
    X /= np.max(X)
    return X, labels
