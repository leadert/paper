#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


torch.cuda.set_device(0)

# # 引入attribute向量
# attribute_embedding = np.load("generateData/Attribute_High_Embedding.npy")

# # 引入category向量
# category_embedding = np.load("generateData/Category_Embedding.npy")

# # shop特征矩阵
# shop_matrix = np.concatenate((attribute_embedding, category_embedding), axis=1)
# padLine = np.zeros([1, attribute_embedding.shape[1]+category_embedding.shape[1]])
# shop_matrix = np.insert(shop_matrix, 0, values=padLine, axis=0)      # pad 0
# shop_matrix = torch.from_numpy(shop_matrix)

# In[4]:
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out

# In[30]:
# embed_dim = shop_matrix.shape[1]
# shop_size, shop_embed_size = shop_matrix.shape[0], shop_matrix.shape[1]

class Net(torch.nn.Module):
    def __init__(self, shop_matrix):
        super(Net, self).__init__()
        self.shop_size = shop_matrix.shape[0]
        self.shop_embed_size = shop_matrix.shape[1]
        self.embed_dim = shop_matrix.shape[1]

        self.shop_embedding = nn.Embedding(self.shop_size, self.shop_embed_size, padding_idx=0)
        self.shop_embedding.weight.data.copy_(torch.from_numpy(shop_matrix))
        # self.shop_embedding.weight.requires_grad = False
        
        self.conv1 = GCNConv(self.embed_dim, self.embed_dim)
        self.bn = nn.BatchNorm1d(self.embed_dim)
        self.active = nn.ReLU()
        self.conv2 = GCNConv(self.embed_dim, self.embed_dim)

        self.lstm1 = nn.LSTM(input_size=self.embed_dim, hidden_size=self.embed_dim, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.embed_dim, hidden_size=self.embed_dim, batch_first=True)
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(2*self.embed_dim),
            nn.Linear(2*self.embed_dim, self.embed_dim),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(2*self.embed_dim),
            nn.Linear(2*self.embed_dim, self.embed_dim),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.BatchNorm1d(2*self.embed_dim),
            nn.Linear(2*self.embed_dim, self.embed_dim),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.BatchNorm1d(2*self.embed_dim),
            nn.Linear(2*self.embed_dim, self.embed_dim),
            nn.ReLU()
        )
        self.fc5 = nn.Sequential(
            nn.BatchNorm1d(self.embed_dim),
            nn.Linear(self.embed_dim, 5)
        )
        
        # attention1参数
        self.w_omega = nn.Parameter(torch.Tensor(self.embed_dim, self.embed_dim))
        self.u_omega = nn.Parameter(torch.Tensor(self.embed_dim, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        
        # attention2参数
        self.w_omega2 = nn.Parameter(torch.Tensor(self.embed_dim, self.embed_dim))
        self.u_omega2 = nn.Parameter(torch.Tensor(self.embed_dim, 1))
        nn.init.uniform_(self.w_omega2, -0.2, 0.2)
        nn.init.uniform_(self.u_omega2, -0.2, 0.2)
        
        # attention3参数
        self.w_omega3 = nn.Parameter(torch.Tensor(self.embed_dim, self.embed_dim))
        self.u_omega3 = nn.Parameter(torch.Tensor(self.embed_dim, 1))
        nn.init.uniform_(self.w_omega3, -0.1, 0.1)
        nn.init.uniform_(self.u_omega3, -0.1, 0.1)
       
    def attention_net(self, x):   #x:[batch_size, seq_len, hidden_dim]
        u = torch.tanh(torch.matmul(x, self.w_omega))          #[batch, seq_len, hidden_dim]
        att = torch.matmul(u, self.u_omega)                           #[batch, seq_len, 1]
        att_score = F.softmax(att, dim=1)
        scored_x = x * att_score                           #[batch, seq_len, hidden_dim]
        feat = torch.sum(scored_x, dim=1)          #[batch, hidden_dim]
        return feat
    
    def attention_net2(self, x):   
        u = torch.tanh(torch.matmul(x, self.w_omega2))
        att = torch.matmul(u, self.u_omega2)
        att_score = F.softmax(att, dim=1)
        scored_x = x * att_score
        feat = torch.sum(scored_x, dim=1) 
        return feat
    
    def attention_net3(self, x):   
        u = torch.tanh(torch.matmul(x, self.w_omega3))
        att = torch.matmul(u, self.u_omega3)
        att_score = F.softmax(att, dim=1)
        scored_x = x * att_score
        feat = torch.sum(scored_x, dim=1) 
        return feat
    
    def forward(self, shop_idxs, data, userId, poiId, userHistory, userHistoryLength):
        x, edge_index = data.x, data.edge_index
        x = self.shop_embedding(shop_idxs)
        x = self.bn(x)
        x = self.conv1(x, edge_index)
        x = self.active(x)
        x = self.conv2(x, edge_index)
        x = self.active(x)

        currentPOI = self.shop_embedding(poiId)
        # 不等长lstm的batch输入处理，batch按照长度由大到小排序
        userHistory_lengths, idx = userHistoryLength.sort(0, descending=True)
        _, un_idx = torch.sort(idx, dim=0)
        userHistory = userHistory[idx]
        userHistory_POI = self.shop_embedding(userHistory)
        userHistory_POI_packed_input = pack_padded_sequence(input=userHistory_POI, lengths=userHistory_lengths, batch_first=True)
        # hidden_state集合；userHistory_POI_packed_out是[batch_size*seq_len,embed_dim]的tensor;_[0]是代表每个seq最后一个hiddenState构成的tensor,[1, batch_size, embedding]
        userHistory_POI_packed_out, (h_n, c_n) = self.lstm1(userHistory_POI_packed_input)
        # userHistory_POI_packed_out解压，仍按照长度由大到小排序, [batch_size,seq_len, embed_dim];_是一个tensor，每个元素代表每个seq的真实长度
        userHistory_POI_out, _ = pad_packed_sequence(userHistory_POI_packed_out, batch_first=True)
        # userHistory_POI_out恢复正常顺序，[batch_size,seq_len, embed_dim]，其后有补0
        userHistory_POI_out = torch.index_select(userHistory_POI_out, 0, un_idx)

        # 使用attention机制对seq的所有hidden_state加权, long_userPref_POI[batch_size, embed_dim]
        userAvePOIPref_attn = self.attention_net(userHistory_POI_out)
        userAvePOIPref = torch.cat([userAvePOIPref_attn, currentPOI], -1)                #1
        # 每个seq的最后一个hidden_state构成的集合,short_userPref_POI[batch_size, embed_dim]
        userShortPOIPref = torch.index_select(h_n, 1, un_idx)
        userShortPOIPref = torch.squeeze(userShortPOIPref, 0)
        userShortPOIPref = torch.cat([userShortPOIPref, currentPOI], -1)                      #2
        
        
        region_embedding = nn.Embedding(self.shop_size, self.shop_embed_size, padding_idx=0).cuda()
        region_embedding.weight.data.copy_(x)
        currentPOIRegion = region_embedding(poiId)
        userHistory_Region = region_embedding(userHistory)
        userHistory_Region_packed_input = pack_padded_sequence(input=userHistory_Region, lengths=userHistory_lengths, batch_first=True)
        userHistory_Region_packed_out, (h_nr, c_nr) = self.lstm2(userHistory_Region_packed_input)
        userHistory_Region_out, _ = pad_packed_sequence(userHistory_Region_packed_out, batch_first=True)
        userHistory_Region_out = torch.index_select(userHistory_Region_out, 0, un_idx)
        userAveRegionPref_attn = self.attention_net2(userHistory_Region_out)
        userAveRegionPref = torch.cat([userAveRegionPref_attn, currentPOIRegion], -1)           #3
        userShortRegionPref = torch.index_select(h_nr, 1, un_idx)
        userShortRegionPref = torch.squeeze(userShortRegionPref, 0)
        userShortRegionPref = torch.cat([userShortRegionPref, currentPOIRegion], -1)           #4
        
        userAvePOIPref = self.fc1(userAvePOIPref)
        userShortPOIPref = self.fc2(userShortPOIPref)
        userAveRegionPref = self.fc3(userAveRegionPref)
        userShortRegionPref = self.fc4(userShortRegionPref)
        
        userPref = torch.stack([userAvePOIPref, userShortPOIPref, userAveRegionPref, userShortRegionPref], 1)
        userPref = self.attention_net3(userPref)
        output = self.fc5(userPref)
        return output

# In[ ]:


# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=opt.weight_decay)


# In[ ]:




