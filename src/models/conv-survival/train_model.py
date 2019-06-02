from skorch import NeuralNet
from skorch.callbacks import EarlyStopping
from torch.nn import MSELoss
from torch.optim import SGD
import numpy as np

import sys
sys.path.append('..')
from helpers import load_data_in_chunks, save_model
from model import Net
from RelativeEntropyLoss import RelativeEntropyLoss

(Xs, Ys) = load_data_in_chunks('survival', 'train', chunk_size=5)
Xs = Xs.astype(np.float32)
Ys = Ys.astype(np.float32)

regr = NeuralNet(
    Net,
    max_epochs=10000000000,
    batch_size=100,
    iterator_train__shuffle=True,
    criterion=RelativeEntropyLoss,
    optimizer=SGD,
    optimizer__lr=1e-5,
    optimizer__momentum=0.9,
    optimizer__nesterov=True,
    optimizer__dampening=0,
    verbose=5,
    callbacks=[('early_stop', EarlyStopping())]
)

regr.fit(Xs, Ys)

save_model(regr, 'conv-survival')
