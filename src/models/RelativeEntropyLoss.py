import torch
from torch.nn.modules.loss import _Loss

class RelativeEntropyLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='none'):
        super(RelativeEntropyLoss, self).__init__(size_average, reduce, reduction)
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, input, target):
        reduction = self.reduction
        (O, T) = (input, target)
        ret_a = (1 + T) * torch.log((1 + T) / (1 + O))
        ret_b = (1 - T) * torch.log((1 - T) / (1 - O))
        ret = 0.5 * (ret_a + ret_b)
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
        return ret
