from model import AdaGAE
import torch
import data_loader as loader
from metrics import ClusteringMetrics
import scipy.io as scio
import warnings
import numpy as np

warnings.filterwarnings('ignore')


dataset = loader.UMIST
[data, labels] = loader.load_data(dataset)
# [data, labels, links] = loader.load_cora()
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
X = torch.Tensor(data).to(device)
input_dim = data.shape[1]
layers = None
if dataset is loader.USPS:
    layers = [input_dim, 128, 64]
elif dataset is loader.TOY_THREE_RINGS:
    layers = [input_dim, 3, 2]
elif dataset is loader.SEGMENT:
    layers = [input_dim, 10, 7]
elif dataset is loader.GLASS:
    layers = [input_dim, 7, 5]
elif dataset is loader.CORA:
    layers = [input_dim, 1600, 400]
else:
    layers = [input_dim, 256, 64]
# for lam in [0.001, 0.01, 0.1, 1, 10, 100, 512, 1024]:
accs = [];
nmis = [];
for lam in np.power(2.0, np.array(range(-10, 10, 2))):
# for lam in [10**-2]:
    # for neighbors in [5]:
    for neighbors in [5]:
        print('-----lambda={}, neighbors={}'.format(lam, neighbors))
        gae = AdaGAE(X, labels, layers=layers, num_neighbors=neighbors, lam=lam, max_iter=50, max_epoch=10,
                     update=True, learning_rate=5*10**-3, inc_neighbors=5, device=device)
        acc, nmi = gae.run()
    accs.append(acc)
    nmis.append(nmi)
print(accs)
print(nmis)