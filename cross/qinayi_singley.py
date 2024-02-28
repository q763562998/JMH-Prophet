import json
import torch

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv, TopKPooling, GraphUNet, global_mean_pool, GATConv
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
from torch_geometric.nn import GCN
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from torch_geometric.nn import GCNConv,global_max_pool, GATConv, GATv2Conv,GraphSAGE,SAGEConv


from torch_geometric.nn import GraphConv,global_mean_pool,global_add_pool
from torch_geometric.loader import DataLoader

import numpy as np
import networkx as nx
import os
import math
from torch_geometric.transforms import NormalizeFeatures


import random

import numpy as np
import torch
from torch_geometric.nn import GCN

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from torch_geometric.nn import GCNConv,global_max_pool, GATConv, GATv2Conv,GraphSAGE,SAGEConv


from torch_geometric.nn import GraphConv,global_mean_pool,global_add_pool
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import networkx as nx
import numpy as np 
import torch
from torch_geometric.data import Data

from torch.utils.data import DataLoader
import torch_geometric
import random
from torch.utils.data import DataLoader, Subset
from datetime import datetime





json_path="/home/zhs/predict/expertcode/data/tn.json"
with open(json_path, "r") as file:
    data = json.load(file)


with open("/home/zhs/predict/model/project_keywords.json", "r") as file:
    project_keywords = json.load(file)



all_project_keys = list(project_keywords.keys())



for p_key in all_project_keys:
    pname=p_key

    currentDateAndTime = datetime.now()
    print(currentDateAndTime)
    ori_key_list=[]
    ori_body_list=[]
    ori_len_list=[]
    ori_tree_list=[]
    ori_cfg_list=[]
    ori_lables_list=[]

    
    
    
    tree_batch = torch.load('/home/zhs/predict/expertcode/data/all_tree.pt')

    cfg_batch = torch.load('/home/zhs/predict/expertcode/data/all_cfg.pt')

    

    len(cfg_batch)
    len(tree_batch)

    time=1
    
    for key in data.keys():
        
        value = json.loads(data[key])
        key = key.split("#")[0]
        
        if  not any(keyword in key for keyword in project_keywords[pname]):
            continue    
        
        

        
        size=value.get("len")
        body=value.get("indexList")
        lables=value.get("lables")
        
        if lables==None :
            continue
    
        d=0
        
        for tree in tree_batch:

            if tree.y==key:
                d=1
                
                t=tree
                continue
        
        
        c=0    
        for cfg in cfg_batch:
            if cfg.y==key:
                
                c=1
                
                g=cfg
                continue
        if c==1 & d ==1:
            
            ori_key_list.append(key)
            ori_len_list.append(size)
            ori_body_list.append(body)
            ori_lables_list.append(lables)
            ori_cfg_list.append(g)
            ori_tree_list.append(t)
        time=time-1

    names = ori_body_list
    
    
                    


    
    namelen =ori_len_list

    seed = 20
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



    
    
    




    labels=list(map(float, ori_lables_list))

    

    labels = np.array(labels)
    
    normalized_labels = (labels - labels.min()) / (labels.max() - labels.min())





    
    log_array = [math.log(x + 1) for x in labels]

    
    max_value = max(log_array)

    
    normalized_array = [x / max_value for x in log_array]


    
    labels=torch.tensor(normalized_labels)
    
    
    labels=normalized_array

    from torch.utils.data import Dataset, DataLoader

    class MultiModalDataset(object):
        def __init__(self, names,namelen, tokns_list, graphs_list, labels,ori_key_list):

            self.names = names
            self.namelen=namelen
            self.tokns_list = tokns_list
            
            self.graphs_list = graphs_list
            
            self.labels = labels
            self.method = ori_key_list
        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.names[idx] , self.namelen[idx] , self.tokns_list[idx] , self.graphs_list[idx] , self.labels[idx],self.method[idx]





    
    dataset = MultiModalDataset(names,namelen, ori_tree_list, ori_cfg_list, labels,ori_key_list)
    


    def collate_fn(batch):
        names, namelen, tokns_list, graphs_list, labels,ori_key_list = zip(*batch)

        
        names_batch = torch.tensor(names, dtype=torch.long)
        namelen_batch = torch.tensor(namelen, dtype=torch.long)
        tokns_batch = torch_geometric.data.Batch.from_data_list(tokns_list)
        graphs_batch = torch_geometric.data.Batch.from_data_list(graphs_list)
        labels_batch = torch.stack([torch.tensor(l, dtype=torch.double) for l in labels], dim=0)
        method_batch= ori_key_list


        return names_batch, namelen_batch, tokns_batch, graphs_batch, labels_batch,method_batch


    
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GraphConv, TopKPooling, GraphUNet, global_mean_pool, GATConv



    class BOWEncoder(nn.Module):
        def __init__(self, vocab_size, emb_size, hidden_size):
            super(BOWEncoder, self).__init__()
            self.emb_size=emb_size
            self.hidden_size = hidden_size
            self.embedding = nn.Embedding(vocab_size, emb_size)
            self.init_weights()
            
        def init_weights(self):
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            nn.init.constant_(self.embedding.weight[0], 0)
            
        def forward(self, input, input_len=None): 
            batch_size, seq_len =input.size()
            embedded = self.embedding(input)
            embedded= F.dropout(embedded, 0.25, self.training)
            
            
            output_pool = F.max_pool1d(embedded.transpose(1,2), seq_len).squeeze(2) 
            encoding = output_pool 
            return encoding


    
    class SeqEncoder(nn.Module):
        def __init__(self, vocab_size, emb_size, hidden_size, n_layers=1):
            super(SeqEncoder, self).__init__()
            self.emb_size = emb_size
            self.hidden_size = hidden_size
            self.n_layers = n_layers
            self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
            self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
            self.init_weights()
            
        def init_weights(self):
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            nn.init.constant_(self.embedding.weight[0], 0)
            for name, param in self.lstm.named_parameters():
                if 'weight' in name or 'bias' in name: 
                    param.data.uniform_(-0.1, 0.1)

        def forward(self, inputs, input_lens=None): 
            
            batch_size, seq_len=inputs.size()
            inputs = self.embedding(inputs)  
            inputs = F.dropout(inputs, 0.25, self.training)
            
            if input_lens is not None:
                
                input_lens_sorted, indices = input_lens.sort(descending=True)
                inputs_sorted = inputs.index_select(0, indices)        
                inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)
                
            hids, (h_n, c_n) = self.lstm(inputs) 
            
            if input_lens is not None: 
                
                _, inv_indices = indices.sort()
                hids, lens = pad_packed_sequence(hids, batch_first=True)   
                hids = F.dropout(hids, p=0.25, training=self.training)
                hids = hids.index_select(0, inv_indices)
                h_n = h_n.index_select(1, inv_indices)
            h_n = h_n.view(self.n_layers, 2, batch_size, self.hidden_size) 
            
            h_n = h_n[-1] 
            
            
            encoding = h_n.view(batch_size,-1) 
            return encoding 




    
    class MyModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(MyModel, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)  
            self.relu = nn.ReLU()  
            self.fc2 = nn.Linear(hidden_size, output_size)  

        def forward(self, x):
            out = self.fc1(x.float())
            out = self.relu(out)
            out = self.fc2(out)
            return out


    class MultiModel(torch.nn.Module):
        def __init__(self, in_channels, out_channels, hidden_channels, num_layers):
            super(MultiModel, self).__init__()
            
            pool_ratios = 0.6
            
            self.tree_conv1 = GraphUNet(13,out_channels ,hidden_channels,num_layers,pool_ratios)
            
            self.graph_conv1 = GraphUNet(20, out_channels, hidden_channels, num_layers, pool_ratios)
            

            
            vocab_size = 200000
            emb_size = 512
            hidden_size = 256
            
            self.encoder=BOWEncoder(vocab_size, emb_size, hidden_size)
            
            
            
            self.my_encoder=MyModel(16,hidden_size,emb_size)


            self.lin1=torch.nn.Linear(hidden_channels,out_channels)
            self.lin2=torch.nn.Linear(out_channels,512)
            self.hidden=512
            
    
            
            
            
            self.tree_gat1 = GATConv(hidden_channels, hidden_channels, heads=1, concat=False)
            self.graph_gat1 = GATConv(hidden_channels, hidden_channels, heads=1, concat=False)


            self.lin3 = torch.nn.Linear(hidden_channels, 512)

            lstm_dims=256
            n_hidden=512
            emb_size=512


            self.w_name = nn.Linear(2*lstm_dims, n_hidden)
            self.w_tok = nn.Linear(emb_size, n_hidden)
            self.w_graphseq=nn.Linear(2*lstm_dims, n_hidden)  
            self.w_desc = nn.Linear(2*lstm_dims, n_hidden)
            
            self.w_atten = nn.Linear(n_hidden, 1)
            
            self.fuse = nn.Linear(n_hidden*3,n_hidden)
            
            self.fuse4 = nn.Linear(n_hidden*4,n_hidden)
            self.init_weights()

    
            self.fc1 = nn.Linear(512, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 1)
            
            self.fc4 = nn.Linear(512, 1)

            self.drop1=torch.nn.Dropout()
            self.drop2=torch.nn.Dropout()
        def init_weights(self):
            for m in [self.w_name, self.w_tok, self.w_graphseq, self.w_desc, self.w_atten, self.fuse]:        
                m.weight.data.uniform_(-0.1, 0.1)
                nn.init.constant_(m.bias, 0.)


        
        def code_encoding(self, name_repr, tok_repr, graphseq_repr):

            batch_size=name_repr.shape[0]
        
            
            
            
            
            
            
            
            

    

            
            
            name_feat_hidden = self.w_name(name_repr).reshape(-1, self.hidden)
            tok_feat_hidden = self.w_tok(tok_repr).reshape(-1, self.hidden)
            graphseq_feat_hidden = self.w_graphseq(graphseq_repr).reshape(-1, self.hidden)
            
            name_attn_tanh = torch.tanh(name_feat_hidden)
            name_attn_scalar = self.w_atten(F.dropout(name_attn_tanh, 0.25 ,self.training).reshape(-1, self.hidden))            
            
            tok_attn_tanh = torch.tanh(tok_feat_hidden)
            tok_attn_scalar = self.w_atten(F.dropout(tok_attn_tanh, 0.25 ,self.training).reshape(-1, self.hidden))
            
            graphseq_attn_tanh = torch.tanh(graphseq_feat_hidden)
            graphseq_attn_scalar = self.w_atten(F.dropout(graphseq_attn_tanh, 0.25 ,self.training).reshape(-1, self.hidden))
            
            



            
            attn_cat = torch.cat([name_attn_scalar, tok_attn_scalar, graphseq_attn_scalar], 1)
            
            
            
            atten_weight = torch.sigmoid(attn_cat)


            
            name_feat_atten = torch.bmm(atten_weight[:,0].reshape(batch_size, 1, 1),name_repr.reshape(batch_size, 1, self.hidden))     
            tok_feat_atten = torch.bmm(atten_weight[:,0].reshape(batch_size, 1, 1),tok_repr.reshape(batch_size, 1, self.hidden))
            graphseq_feat_atten = torch.bmm(atten_weight[:,0].reshape(batch_size, 1, 1),graphseq_repr.reshape(batch_size, 1, self.hidden))

            
            
            
            
            
            cat_atten_repr = torch.cat((name_feat_atten, tok_feat_atten, graphseq_feat_atten), 2)
            code_repr = torch.tanh(self.fuse(F.dropout(cat_atten_repr, 0.25, training=self.training))).reshape(-1,self.hidden)
            return code_repr 

        def four_modal_fusion(self, name_repr, tok_repr, graphseq_repr, audio_repr):
            batch_size = name_repr.shape[0]

            
            name_feat_hidden = self.w_name(name_repr).reshape(-1, self.hidden)
            tok_feat_hidden = self.w_tok(tok_repr).reshape(-1, self.hidden)
            graphseq_feat_hidden = self.w_graphseq(graphseq_repr).reshape(-1, self.hidden)
            audio_feat_hidden = self.w_audio(audio_repr).reshape(-1, self.hidden)

            name_attn_tanh = torch.tanh(name_feat_hidden)
            name_attn_scalar = self.w_atten(F.dropout(name_attn_tanh, 0.25, self.training).reshape(-1, self.hidden))

            tok_attn_tanh = torch.tanh(tok_feat_hidden)
            tok_attn_scalar = self.w_atten(F.dropout(tok_attn_tanh, 0.25, self.training).reshape(-1, self.hidden))

            graphseq_attn_tanh = torch.tanh(graphseq_feat_hidden)
            graphseq_attn_scalar = self.w_atten(F.dropout(graphseq_attn_tanh, 0.25, self.training).reshape(-1, self.hidden))

            audio_attn_tanh = torch.tanh(audio_feat_hidden)
            audio_attn_scalar = self.w_atten(F.dropout(audio_attn_tanh, 0.25, self.training).reshape(-1, self.hidden))

            
            attn_cat = torch.cat([name_attn_scalar, tok_attn_scalar, graphseq_attn_scalar, audio_attn_scalar], 1)
            atten_weight = F.softmax(attn_cat, dim=1)

            
            name_feat_atten = torch.bmm(atten_weight[:, 0].reshape(batch_size, 1, 1), name_repr.reshape(batch_size, 1, self.hidden))
            tok_feat_atten = torch.bmm(atten_weight[:, 0].reshape(batch_size, 1, 1), tok_repr.reshape(batch_size, 1, self.hidden))
            graphseq_feat_atten = torch.bmm(atten_weight[:, 0].reshape(batch_size, 1, 1), graphseq_repr.reshape(batch_size, 1, self.hidden))
            audio_feat_atten = torch.bmm(atten_weight[:, 0].reshape(batch_size, 1, 1), audio_repr.reshape(batch_size, 1, self.hidden))

            
            cat_atten_repr = torch.cat((name_feat_atten, tok_feat_atten, graphseq_feat_atten, audio_feat_atten), 2)

            
            code_repr = torch.tanh(self.fuse(F.dropout(cat_atten_repr, 0.25, training=self.training))).reshape(-1, self.hidden)
            return code_repr

        def forward(self, name, namelen,tree_x,tree_edge_index,tree_batch,graph_x,graph_edge_index,graph_batch):
            

            name_repr=self.encoder(name,namelen) 

            
            




            
            tree_x=self.tree_conv1(tree_x,tree_edge_index,tree_batch)
            tree_x=self.tree_gat1(tree_x,tree_edge_index)
            tree_x=F.relu(tree_x)
            
            tree_x=self.drop1(tree_x)

            
            tree_x=self.lin3(tree_x)
            
            
            tree_x=global_mean_pool(tree_x,tree_batch)

            tok_repr = tree_x






        
            graph_x=self.graph_conv1(graph_x,graph_edge_index,graph_batch)
            graph_x=self.graph_gat1(graph_x,graph_edge_index)
            graph_x=F.relu(graph_x)
            
            graph_x=self.drop2(graph_x)

            
            graph_x=self.lin3(graph_x)
            graph_x=global_mean_pool(graph_x,graph_batch)
            
            graphseq_repr = graph_x
        

            
            
            x=self.code_encoding(name_repr,tok_repr,graphseq_repr)

            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))

            
            return torch.tanh(self.fc3(x))  
            
            

        def my_sigm(x):
            
            return 1 / (1+torch.exp(-2*x))
        


    from scipy.stats import pearsonr
    import matplotlib.pyplot as plt
        

    def train(train_loader):
        
        model.train()
        optimizer.zero_grad(device)

        total_loss = 0
        for names_batch, namelen_batch, tokns_batch, graphs_batch, labels_batch,method in train_loader:
            
            
            
            names_batch=names_batch.to(device)
            namelen_batch=namelen_batch.to(device)
            tokns_batch=tokns_batch.to(device)
            graphs_batch=graphs_batch.to(device)
            labels_batch=labels_batch.to(device)
            
            
            
            tokns_batch.x=tokns_batch.x.to(device)
            tokns_batch.edge_index=tokns_batch.edge_index.to(device)
            tokns_batch.batch=tokns_batch.batch.to(device)
            
            

            optimizer.zero_grad()
            output=model(names_batch,namelen_batch,
                tokns_batch.x,tokns_batch.edge_index,tokns_batch.batch,
                graphs_batch.x,graphs_batch.edge_index,graphs_batch.batch)
            
            
            
            
            y = labels_batch.view(-1, 1).float()
            loss = F.mse_loss(output, y)
            
            
            loss.backward()
            optimizer.step()
            
            total_loss += float(loss) * len(namelen_batch)
            

        
        return total_loss / len(train_loader.dataset)


    @torch.no_grad()
    def test(loader,my_size):
        model.eval()

        total_loss = 0
        preds = []
        reals = []
        for names_batch, namelen_batch, tokns_batch, graphs_batch, labels_batch ,method in loader:
            

            names_batch=names_batch.to(device)
            namelen_batch=namelen_batch.to(device)
            
            tokns_batch.x=tokns_batch.x.to(device)
            tokns_batch.edge_index=tokns_batch.edge_index.to(device)
            tokns_batch.batch=tokns_batch.batch.to(device)

            graphs_batch.x=graphs_batch.x.to(device)
            graphs_batch.edge_index=graphs_batch.edge_index.to(device)
            graphs_batch.batch=graphs_batch.batch.to(device)

            labels_batch=labels_batch.to(device)
            
            output=model(names_batch,namelen_batch,
                tokns_batch.x,tokns_batch.edge_index,tokns_batch.batch,
                graphs_batch.x,graphs_batch.edge_index,graphs_batch.batch)
            
            y = labels_batch.view(-1, 1).float()

        

            
            loss = F.mse_loss(output, y)

            total_loss += float(loss) * len(namelen_batch)
            

            pred=output.detach().cpu().numpy()
            

            real=y.detach().cpu().numpy()
            
            preds = np.concatenate((preds,pred.flatten() ), axis=0)
            reals = np.concatenate((reals, real.flatten()), axis=0)
            
        
        return total_loss / len(loader.dataset) ,pearsonr(preds,reals)[0],preds,reals




    import random
    from torch.utils.data import DataLoader, Subset

    

    
    indices = list(range(len(dataset)))
    random.shuffle(indices) 

    
    k_folds = 5

    fold_size = len(dataset) // k_folds  

    
    all_accuracy = []



    for lun in range(1):

        currentDateAndTime = datetime.now()
 
        for fold in range(k_folds):

            
            currentDateAndTime = datetime.now()
            test_indices = indices[fold * fold_size : (fold + 1) * fold_size]
            train_indices = indices[:fold * fold_size] + indices[(fold + 1) * fold_size:]
            
            
            train_dataset = Subset(dataset, train_indices)
            test_dataset = Subset(dataset, test_indices)

        
        
        
            
            print("---------")
            
            batch_size = 32
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            
        
            
            
            
            
            
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
            out_channels=256
            hidden_channels=256
            num_layers=3
            model = MultiModel(1, out_channels, hidden_channels, num_layers).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            criterion = torch.nn.MSELoss()  
            
            num_epochs=100
            data_collection = []
            for epoch in range(1, num_epochs):
                currentDateAndTime = datetime.now()
                
                loss = train(train_loader)
                
                
                my_size=len(train_loader)
                
                train_loss,pr,pred,real = test(train_loader,my_size)

                my_size=len(test_loader)
                test_loss,prt,predt,realt = test(test_loader,my_size)

            
                
            
                


                
                if epoch % 5 == 0:
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
                        print("index:{}".format(test_indices))
                    data_collection.append((pr, prt, epoch, loss, test_loss))  

                

    
    
