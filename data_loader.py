import scipy.io as scio
import numpy as np

YALE = 'Yale'
UMIST = 'UMIST'
ORL = 'ORL'
COIL20 = 'COIL20'
YALEB_SDSIFT = 'YaleB_DSIFT'
JAFFE = 'JAFFE'
MNIST_DSIFT = 'mnist_DSIFT'
PALM = 'Palm'
USPS = 'USPSdata_20_uni'
TOY_THREE_RINGS = 'three_rings'
MNIST_TEST = 'mnist_test'
YEAST = 'yeast_uni'
SEGMENT = 'segment_uni'
NEWS = '20news_uni'
WEBK = 'WebKB_wisconsin_uni'
TEXT = 'text1_uni'
GLASS = 'glass_uni'
ISOLET = 'Isolet'
CORA = 'cora'


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
