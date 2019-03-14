import torch
import math
from torch.nn.modules.loss import _Loss

class CustomLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='none'):
        super(CustomLoss, self).__init__(size_average, reduce, reduction)
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, input, target):
        (theta, thetahat) = (target.squeeze(), input.squeeze())
        term1 = torch.log(1 + (1 - thetahat / theta) ** 2)
        term2 = torch.log(math.exp(1) + (theta / thetahat))
        print(theta / thetahat)
        return torch.sum(term1 * term2)
