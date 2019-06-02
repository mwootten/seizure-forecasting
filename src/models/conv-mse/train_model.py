from skorch import NeuralNet
from skorch.callbacks import EarlyStopping
from torch.nn import MSELoss
from torch.optim import SGD
import numpy as np

import sys
sys.path.append('..')
from helpers import load_data_in_chunks, save_model
from model import Net
from CustomLoss import CustomLoss

(Xs, Ys) = load_data_in_chunks('basic', 'train', chunk_size=5)
Xs = Xs.astype(np.float32)
Ys = Ys.astype(np.float32)

regr = NeuralNet(
    Net,
    max_epochs=10000000000,
    batch_size=100,
    iterator_train__shuffle=True,
    criterion=CustomLoss,
    optimizer=SGD,
    optimizer__lr=1e-5,
    optimizer__momentum=0.9,
    optimizer__nesterov=True,
    optimizer__dampening=0,
    verbose=5,
    callbacks=[('early_stop', EarlyStopping())]
)
regr.fit(Xs, Ys / 5000)

save_model(regr, 'conv-mse')
