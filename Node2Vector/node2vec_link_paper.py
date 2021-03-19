import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import train_test_split_edges
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from numpy.random import randint
from sklearn.preprocessing import StandardScaler
from sklearn import metrics, model_selection, pipeline

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
data = dataset[0]
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)


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


def test():
    model.eval()
    z = model()

    train_z = torch.tensor(()).to(device)
    test_z = torch.tensor(()).to(device)

    for a, b in zip(train_pos_edge_index[0], train_pos_edge_index[1]):
        new_z = torch.unsqueeze(z[a] * z[b], 0)
        train_z = torch.cat((train_z, new_z), 0)
    for _ in range(len(train_pos_edge_index[0])):
        c_idx = randint(len(z))
        d_idx = randint(len(z))
        if data.train_neg_adj_mask[c_idx][d_idx]:
            new_z = torch.unsqueeze(z[c_idx] * z[d_idx], 0)
            train_z = torch.cat((train_z, new_z), 0)

    labels = np.zeros(len(train_z))
    labels[: len(train_pos_edge_index[0])] = 1
   
    # Linear classifier
    scaler = StandardScaler()
    lin_clf = LogisticRegression(C=1)
    clf = pipeline.make_pipeline(scaler, lin_clf)
    clf.fit(train_z.detach().cpu().numpy(), labels)
    # clf = LogisticRegression(solver='lbfgs', multi_class='auto')\
    #     .fit(train_z.detach().cpu().numpy(), labels)

    for a, b in zip(test_pos_edge_index[0], test_pos_edge_index[1]):
        try:
            new_z = torch.unsqueeze(z[a] * z[b], 0)
        except:
            print(a, b, len(z))
        test_z = torch.cat((test_z, new_z), 0)

    for a, b in zip(test_neg_edge_index[0], test_neg_edge_index[1]):
        new_z = torch.unsqueeze(z[a] * z[b], 0)
        test_z = torch.cat((test_z, new_z), 0)
    test_labels = np.zeros(len(test_z))
    test_labels[: len(test_pos_edge_index[0])] = 1
   
    pred = clf.predict(test_z.detach().cpu().numpy())

    return roc_auc_score(test_labels, pred), average_precision_score(test_labels, pred)


auc_list = []
ap_list = []
for _ in range(5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Node2Vec(data.train_pos_edge_index, embedding_dim=128, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)
    node = data.x.to(device)
    train_pos_edge_index = data.train_pos_edge_index.to(device)
    test_pos_edge_index = data.test_pos_edge_index.to(device)
    test_neg_edge_index = data.test_neg_edge_index.to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    for epoch in range(1, 101):
        loss = train()
        # print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))
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
