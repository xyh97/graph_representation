import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, DeepGraphInfomax
# from torch_geometric.utils import train_test_split_edges

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora',
                    choices=['Cora', 'CiteSeer', 'PubMed'])
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()
dataset = args.dataset
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

dataset = Planetoid(path, dataset)


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


def train():
    model.train()
    optimizer.zero_grad()
    pos_z, neg_z, summary = model(data.x, data.edge_index)
    loss = model.loss(pos_z, neg_z, summary)
    loss.backward()
    optimizer.step()
    return loss.item()


def test():
    model.eval()
    z, _, _ = model(data.x, data.edge_index)
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                     z[data.test_mask], data.y[data.test_mask], max_iter=150)
    return acc


test_acc = []
best_acc = 0
for _ in range(5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepGraphInfomax(
        hidden_channels=512, encoder=Encoder(dataset.num_features, 512),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, 301):
        loss = train()
        # print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))
    acc = test()
    test_acc.append(acc)
    print('Accuracy: {:.4f}'.format(acc))
    if best_acc < acc:
        best_acc = acc
        if not os.path.isdir('../checkpoint'):
            os.makedirs('../checkpoint')
        torch.save(model.state_dict(), os.path.join('../checkpoint', '{}.pth'.format(args.dataset)))
test_acc = np.array(test_acc)
print("Node classification for {}, acc mean {:.4f}, acc std {:.4f}".format(args.dataset,
                                                                         np.mean(test_acc), np.std(test_acc, ddof=1)))
