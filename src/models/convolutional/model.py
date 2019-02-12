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

    def __init__(self, sizes=[643, 160, 40, 10, 2]):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, [2,1])
        self.conv2 = nn.Conv2d(1, 1, [2,1])
        self.conv3 = nn.Conv2d(1, 1, [2,1])
        self.conv4 = nn.Conv2d(1, 1, [2,1])
        self.conv5 = nn.Conv2d(1, 1, [2,1])
        # Convolutional to linear neuron
        self.fc1 = nn.Linear(sizes[0], sizes[1])
        self.fc2 = nn.Linear(sizes[1], sizes[2])
        self.fc3 = nn.Linear(sizes[2], sizes[3])
        self.fc4 = nn.Linear(sizes[3], sizes[4])
        self.fc5 = nn.Linear(sizes[4], 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.max_pool2d(F.relu(self.conv1(x)), [1,1])
        x = F.max_pool2d(F.relu(self.conv2(x)), [1,1])
        x = F.max_pool2d(F.relu(self.conv3(x)), [1,1])
        x = F.max_pool2d(F.relu(self.conv4(x)), [1,1])
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
