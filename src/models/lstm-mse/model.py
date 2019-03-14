import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm1 = nn.LSTM(input_size=624, hidden_size=10, num_layers=1)
        self.fc1 = nn.Linear(10,1)
        self.fc2 = nn.Linear(5,1)

    def forward(self, x):
        batch = len(x)
        x = x.view([5,batch,624])
        x,hn = self.lstm1(x)
        x = x.view([5*batch,10])
        x = torch.tanh(self.fc1(x))
        x = x.view(batch,5)
        x = torch.relu(self.fc2(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
