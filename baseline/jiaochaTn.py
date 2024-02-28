import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import barabasi_albert_graph
import networkx as nx
import pickle as pkl
import random
import numpy as np
import torch_geometric.transforms as T
from typing import Optional, Callable


import numpy as np
import networkx as nx
import os
import math
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import barabasi_albert_graph
import networkx as nx
import pickle as pkl
import random
import numpy as np
import torch_geometric.transforms as T
from typing import Optional, Callable


import numpy as np
import networkx as nx
import os
import math
import numpy as np
from torch_geometric.transforms import NormalizeFeatures


import random

import numpy as np
from torch_geometric.nn import GCN
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from torch_geometric.nn import GCNConv,global_max_pool, GATConv, GATv2Conv,GraphSAGE,SAGEConv


from torch_geometric.nn import GraphConv,global_mean_pool,global_add_pool
from torch_geometric.loader import DataLoader
import torch
from datetime import datetime
import pickle




zhs_lables=[]
class H2_FATree_nf2(InMemoryDataset):
    def __init__(self,file_names,transform: Optional[Callable] = None,pre_filter: Optional[Callable] = None,):
        super().__init__('.',transform,pre_filter)
        random.seed(10)
        data_list = []
        
        
        
        
        
            
        
        for name in file_names:
            
            lab = np.load("/home/zhs/gnn/graphs/"+name+"/labels.npy")
            labels = dict()
            
            for i,j in lab:
                j = j+1
                labels[int(i)] = math.log(j)
            
            max_val = max(list(labels.values()))
            for i,j in labels.items():
                labels[i] = j/max_val
            
            
            with open('lable_set.pkl', 'wb') as f:
                pickle.dump(labels, f)


            
            
            
            
            


            path = "/home/zhs/gnn/graphs/"+name+"/"
            for i in os.listdir(path):
                if i[-2:] == "pk":
                    g = nx.read_gpickle(path+i)
                    edge_index = torch.tensor([[i,j] for i,j in (g.edges())]).T
                    edge_attr = torch.tensor(list(nx.get_edge_attributes(g,"x").values()))
                    x = torch.tensor(list(nx.get_node_attributes(g,"x").values())).double()
                    label = labels[int(i[:-3])]
                    
                    data = Data(x=x, edge_index=edge_index, y=torch.tensor([label]).float(),edge_attr=edge_attr)
                    zhs_lables.append(data.y)
                    unique_id = path+i
                    data.unique_id = unique_id
                    data_list.append(data)
        
        
        self.data, self.slices = self.collate(data_list)


        
        




file_names = ["zhs/tn"]




dataset = H2_FATree_nf2(file_names,transform=NormalizeFeatures())

seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda:3")


print(zhs_lables)
class Net_works(torch.nn.Module):
    def __init__(self,num_features,num_classes):
        super().__init__()
        self.drop1 = torch.nn.Dropout()
        self.conv1 = SAGEConv(num_features, 30)
        self.bn1 = torch.nn.BatchNorm1d(30)
        
        
        self.drop2 = torch.nn.Dropout()
        self.conv2 = SAGEConv(30, 30)
        self.bn2 = torch.nn.BatchNorm1d(30)
        
        
        self.drop3 = torch.nn.Dropout()
        self.conv3 = SAGEConv(30, 30)
        self.bn3 = torch.nn.BatchNorm1d(30)
        self.lin1 = torch.nn.Linear(30,10)
        self.lin2 = torch.nn.Linear(10,1)
        
        
        self.float()

    def forward(self,x,edge_index,edge_attr,batch):
        x = self.drop1(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = self.drop2(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = self.drop3(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        
        x = global_max_pool(x,batch)
        
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        
        
        return my_sigm(x)

def my_sigm(x):
    
    return 1 / (1+torch.exp(-2*x))



class Net_top(torch.nn.Module):
    def __init__(self,num_features,num_classes):
        super().__init__()
        
        
        self.conv1 = GraphConv(num_features, 40)
        self.bn1 = torch.nn.BatchNorm1d(40)
        
        
        self.conv2 = GraphConv(40, 40)
        self.bn2 = torch.nn.BatchNorm1d(40)
        
        self.drop3 = torch.nn.Dropout()
        self.conv3 = GraphConv(40,40)
        self.bn3 = torch.nn.BatchNorm1d(40)
        self.lin1 = torch.nn.Linear(40,10)
        self.lin2 = torch.nn.Linear(10,1)
        
        
        self.float()

    def forward(self,x,edge_index,edge_attr,batch):

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        
        x = self.drop3(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        
        x = global_max_pool(x,batch)
        
        x = F.relu(self.lin1(x))
        x  = self.lin2(x)
        
        
        return torch.sigmoid(x)

def my_sigm(x):
    
    return 1 / (1+torch.exp(-2*x))


idx = torch.arange(len(dataset))


import numpy as np




list_example = idx.tolist()


from scipy.stats import pearsonr
import matplotlib.pyplot as plt
    




def train(loader,m,o):   
    m.train()
    o.zero_grad()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        o.zero_grad()
        output = m(data.x.float(), data.edge_index, data.edge_attr.float(), data.batch)
        
        
        loss = F.mse_loss(output,torch.reshape(data.y, (data.y.shape[0],1)))
        
        
        loss.backward()
        o.step()
        total_loss += float(loss) * data.num_graphs
        

    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(train,loader,m):
    m.eval()

    total_loss = 0
    preds = []
    reals = []
    uniid=[]
    all_time=0
    for data in loader:
        data = data.to(device)
        out = m(data.x.float(), data.edge_index, data.edge_attr.float(), data.batch)

        
        
        
        
        loss = F.mse_loss(out,torch.reshape(data.y, (data.y.shape[0],1)))
        
        all_time=all_time+1
        
        
        total_loss += float(loss) * data.num_graphs
        
        
        
        pred = out.detach().cpu().numpy()[:,0]
        
        real = data.y.detach().cpu().numpy()
        preds.extend(pred)
        reals.extend(real)
        uniid.extend(data.detach().unique_id)
    
    return total_loss / len(loader.dataset) ,pearsonr(preds,reals)[0],preds,reals,uniid


def find_sorted_points(p, v,uniidt):
    
    distances = [abs(pred - real) / np.sqrt(2) for pred, real in zip(p, v)]

    
    sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i], reverse=True)
    res=[]
    for val in sorted_indices:
        res.append(val)     
    return res



import random
set_list=[]
def random_batch_sampling(data_array, batch_size):

    
    sampled_indices = []
    ori=data_array
    
    remaining_data = data_array[:]
    fold=0
    


    while len(sampled_indices) < len(data_array):
        
        remaining_indices = set(range(len(data_array))) - set(sampled_indices)
        remaining_indices = list(remaining_indices)
        current_batch_size = min(batch_size, len(remaining_indices))
        
        
        batch_indices = random.sample(remaining_indices, current_batch_size)
        sampled_indices.extend(batch_indices)
        
        
        
        
        model = Net_top(dataset.num_features,1).to(device).float()
        optimizer = torch.optim.Adam(model.parameters(),lr=0.0001) 

        
        print("Batch Indices:", batch_indices)

        sorted_list = sorted(set_list)
        print(sorted_list)        
        
        train_idx=[]
        for vars in ori:
            if vars not in batch_indices:
                train_idx.append(vars)

        
        train_loader = DataLoader(dataset[train_idx],batch_size=32)
        test_loader = DataLoader(dataset[batch_indices],batch_size=32)
        
        fold+=1
        for epoch in range(1, 101):
            loss = train(train_loader,model,optimizer)
            train_loss,pr,pred,real,uniid = test(train_loader,train_loader,model)
            test_loss,prt,predt,realt,uniidt = test(train_loader,test_loader,model)
                
            if epoch % 10 == 0:

                p_=    predt
                v_=    realt
                u_=    uniidt
                res = find_sorted_points(p_, v_,u_)

                

                plt.figure(figsize=(8,4))
                plt.subplot(121)
                plt.scatter(pred,real)
                plt.ylabel("real")
                plt.xlabel("predict")
                plt.title("train")
                plt.subplot(122)
                plt.scatter(predt,realt)
                plt.ylabel("real")
                plt.xlabel("predict")
                plt.title("test")
                plt.show()
                print("corr train",pr)
                print("corr test ",prt)
                    
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Loss: {test_loss:.4f}')
                print("---"*10)

                if epoch==100:
                    print("pred:{}".format(predt))
                    print("\n")
                    print("real:{}".format(realt))
                    print("\n")
                    print("index:{}".format(batch_indices))
    
    sampled_data = [data_array[i] for i in sampled_indices]
    
    return sampled_data


idx = torch.arange(len(list_example))
data=idx
batch_size = int(len(list_example) * 0.2)

for lun in range(10):
    print("tn")
    print("====================")
    currentDateAndTime = datetime.now()
    print("The current date and time is", currentDateAndTime)
    sampled_data = random_batch_sampling(list_example, batch_size)

