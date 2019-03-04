import torch
import torch.nn as nn
import torch.nn.functional as F

# Adapted from the following tutorial:
#   Title: Neural Networks Tutorial
#   Author: Chintala, S
#   Date: 9/14/2017
#   Code version: 1.0
#   Source: http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 500, [3,1])
        self.conv2 = nn.Conv2d(500, 10, [3,1])
        # Convolutional to linear neuron
        self.fc1 = nn.Linear(643 * 10, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.max_pool2d(F.relu(self.conv1(x)), [1,1])
        x = F.max_pool2d(F.relu(self.conv2(x)), [1,1])
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
