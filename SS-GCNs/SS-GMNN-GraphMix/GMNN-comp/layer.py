import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):

    def __init__(self, opt, adj):
        super(GraphConvolution, self).__init__()
        self.opt = opt
        self.in_size = opt['in']
        self.out_size = opt['out']
        self.adj = adj
        self.weight = Parameter(torch.Tensor(self.in_size, self.out_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_size)
        self.weight.data.uniform_(-stdv, stdv)
    
    def forward(self, x, adj=None):
        if adj is None:
            m0 = torch.spmm(self.adj, x)
            m = torch.mm(m0, self.weight)
        else:
            m0 = torch.spmm(adj, x)
            m = torch.mm(m0, self.weight)
        return m, m0
