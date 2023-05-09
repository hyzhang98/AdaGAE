from model import AdaGAE
import torch
import data_loader as loader
import warnings
import numpy as np

warnings.filterwarnings('ignore')


dataset = loader.BALSAC
[data, labels] = loader.load_data(dataset)
device = torch.device('cpu') # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
X = torch.Tensor(data).to(device)
input_dim = data.shape[1]
layers = None
if dataset is loader.USPS:
    layers = [input_dim, 128, 64]
else:
    layers = [input_dim, 256, 64]
accs = [];
nmis = [];
for lam in np.power(2.0, np.array(range(-10, 10, 2))):
    for neighbors in [5]:
        print('-----lambda={}, neighbors={}'.format(lam, neighbors))
        gae = AdaGAE(X, labels, layers=layers, num_neighbors=neighbors, lam=lam, max_iter=50, max_epoch=10,
                     update=True, learning_rate=5*10**-3, inc_neighbors=5, device=device)
        acc, nmi = gae.run()
    accs.append(acc)
    nmis.append(nmi)
print(accs)
print(nmis)