# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:35:00 2022

@author: Nancy Wang
"""



import pandas as pd
import os
import numpy as np
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from random import random
import warnings


import networkx  as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader

import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling, GCNConv, GATConv, GINEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import to_dense_adj

from sklearn import metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.dec1 = torch.nn.Linear(79, 64)
        self.dec2 = torch.nn.Linear(64, 1)
        # self.dec3 = torch.nn.Linear(128, 32)
        self.line=torch.nn.Linear(1,128)
        
        #self.conv1 = GCNConv(1, 128)
        self.gineconv1 = GINEConv(self.line)
        self.pool1 = TopKPooling(1, ratio=0.8)
        self.conv2 = GCNConv(128, 128)
        #self.gineconv1 = GINEConv(self.conv1)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GATConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)

        self.lin1 = torch.nn.Linear(256, 512)
        self.lin2 = torch.nn.Linear(512, 128)
        self.lin3 = torch.nn.Linear(128, 4)

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        edge_attr = F.relu(self.dec1(edge_attr))
        edge_attr = F.relu(self.dec2(edge_attr))
        # edge_attr = F.relu(self.dec3(edge_attr))

        x = F.relu(self.gineconv1(x=x, edge_index=edge_index, edge_attr=edge_attr))
        x, edge_index, edge_attr1, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, edge_attr2, batch, _, _ = self.pool2(x, edge_index, edge_attr1, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, edge_attr2, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        # print(x.shape)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x     


def train(epoch, train_loader):
    model.train()
    
    loss_all = 0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_loader)

def test(loader):
    model.eval()
    
    correct = 0
    for data in loader:
        #print(model(data).max(dim=1)[1], data.y)
        #print("=============")
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        #print(metrics.classification_report(pred,data.y))
        Prec = metrics.precision_score(pred,data.y,average='weighted')
        Recall = metrics.recall_score(pred,data.y,average='weighted')
        F1 = metrics.f1_score(pred,data.y,average='weighted')
    return correct / len(loader.dataset), Prec, Recall, F1


if __name__ == '__main__':
        for add_number in range(10,110,10):
                print("====================================add benign number is " +str(add_number) +"!!!===================================")
                data_train = torch.load(train_path + 'data_Classification_train_0316_wr0.5.pt')#dataset_savepath
                data_test = torch.load(dataset_savepath + 'data_Classification_test_0520_add_0.7_'+ str(add_number) + '.pt')
                
                print(len(data_train),len(data_test))


       
                data_train_loader = DataLoader(data_train, batch_size = 16)
                data_test_loader = DataLoader(data_test, batch_size = 16)
        
                    
                model = Net().to(device)
        
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) #0.0005
            
                for epoch in range(1, 150):
                        loss = train(epoch, data_train_loader)
                        train_acc,a,b,c = test(data_train_loader)
                        #test_acc= test(data_test_loader)
                        test_acc, Prec, Recall, F1 = test(data_test_loader)
                        print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}, Test Prec: {:.5f},Test Recall: {:.5f},Test F1: {:.5f}'.
                        format(epoch, loss, train_acc,test_acc, Prec, Recall, F1)) #train_acc,Train Acc: {:.5f}, 
                        #print('Epoch: {:03d}, Loss: {:.5f}, Test Acc: {:.5f}'.format(epoch, loss, test_acc, ))
                
                    #torch.save(model, './model/graph_classifier_0402_wr05.pkl')
                  