from skorch import NeuralNet
from torch.nn import MSELoss
from torch.optim import SGD
import numpy as np

import sys
sys.path.append('..')
from helpers import load_data_in_chunks, save_model
from model import Net

(Xs, Ys) = load_data_in_chunks('train', chunk_size=5)
Xs = Xs.astype(np.float32)
Ys = Ys.astype(np.float32)

regr = NeuralNet(
    Net,
    max_epochs=100,
    batch_size=100,
    iterator_train__shuffle=True,
    criterion=MSELoss,
    optimizer=SGD,
    optimizer__lr=1e-4,
    verbose=5
)
regr.fit(Xs, Ys)

save_model(regr, 'convolutional')
