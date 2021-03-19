import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec

# from torch_geometric.utils import train_test_split_edges

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

dataset = Planetoid(path, dataset)


def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test():
    model.eval()
    z = model()
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                        z[data.test_mask], data.y[data.test_mask],
                        max_iter=150)
    return acc


test_acc = []
best_acc = 0
for _ in range(5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)
    # print(dataset[0])
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    for epoch in range(1, 101):
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

