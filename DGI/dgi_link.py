import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, DeepGraphInfomax
from torch_geometric.utils import train_test_split_edges
from sklearn.metrics import roc_auc_score, average_precision_score

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora',
                    choices=['Cora', 'CiteSeer', 'PubMed'])
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()
dataset = args.dataset
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data', dataset)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# dataset = Planetoid(path, dataset)
data = dataset[0]
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x


class InnerProductDecoder(nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper
    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})
    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""
    def forward(self, z, edge_index, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


def train():
    model.train()
    optimizer.zero_grad()
    pos_z, neg_z, summary = model(node, train_pos_edge_index)
    loss = model.loss(pos_z, neg_z, summary)
    loss.backward()
    optimizer.step()
    return loss.item()


def test():
    model.eval()
    z, _, _ = model(node, train_pos_edge_index)
    pos_y = z.new_ones(data.test_pos_edge_index.size(1))
    neg_y = z.new_zeros(data.test_neg_edge_index.size(1))
    y = torch.cat([pos_y, neg_y], dim=0)

    pos_pred = decoder(z, data.test_pos_edge_index, sigmoid=True)
    neg_pred = decoder(z, data.test_neg_edge_index, sigmoid=True)
    pred = torch.cat([pos_pred, neg_pred], dim=0)

    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

    return roc_auc_score(y, pred), average_precision_score(y, pred)


auc_list = []
ap_list = []
for _ in range(5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepGraphInfomax(
        hidden_channels=512, encoder=Encoder(dataset.num_features, 512),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)
    node = data.x.to(device)
    decoder = InnerProductDecoder().to(device)
    train_pos_edge_index = data.train_pos_edge_index.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(1, 401):
        loss = train()
    auc, ap = test()
    auc_list.append(auc)
    ap_list.append(ap)
    print('AUC: {:.4f}, AP: {:.4f}'.format(auc, ap))
    # if best_acc < acc:
    #     best_acc = acc
    #     if not os.path.isdir('../checkpoint'):
    #         os.makedirs('../checkpoint')
    #     torch.save(model.state_dict(), os.path.join('../checkpoint', '{}.pth'.format(args.dataset)))
auc_list = np.array(auc_list)
ap_list = np.array(ap_list)
print("Link prediction for {}, auc mean {:.4f}, auc std {:.4f}, ap mean {:.4f}, ap std {:.4f}".format(
    args.dataset, np.mean(auc_list), np.std(auc_list, ddof=1), np.mean(ap_list), np.std(ap_list, ddof=1)))
