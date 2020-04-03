import os
import numpy as np
from time import time
import torch
import torchsnooper
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv,GatedGraphConv,ChebConv
from utils import Parameters
from gensim.models.poincare import PoincareModel
import process
from pytorch_memlab import profile

torch.cuda.set_device(0)
mask_num=-2**32+1.0
class Net(nn.Module):
    """
    Implementation of my model
    Input: user i[p1, p2, p3,...]
           POI pj
    """
    def __init__(self, poi_size, embedding_matrix, struct_topology, user_history_data_dict):
        super(Net, self).__init__()
        self.verbose = True
        self.use_cuda = True
        self.embedding_dim = Parameters.type_embedding_dim
        self.embedding_matrix = embedding_matrix
        self.struct_topology = struct_topology
        self.hidden_dim = Parameters.HIDDEN_SIZE
        self.linear_size = 30
        self.batch_size = 4
        self.poi_size = poi_size
        self.drop_out = Parameters.DROPOUT_RATE
        self.num_classes = 5
        self.n_epochs = 3
        self.poi_size = embedding_matrix.shape[0]
        self.user_history_data_dict = user_history_data_dict
        self.conv3 = GatedGraphConv(self.hidden_dim, 3)
        # self.conv4 = ChebConv(self.hidden_dim, 56, 2)
        # self.conv5 = ChebConv(16, self.num_classes, 2)

        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.linear_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.linear_size),
            nn.Dropout(self.drop_out)
        )

        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.linear_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.linear_size),
            nn.Dropout(self.drop_out),
            nn.Linear(self.linear_size, self.num_classes)
        )

        self.embed = nn.Embedding(self.poi_size, self.embedding_dim, padding_idx=0)
        self.embed.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.bat_nor_embed = nn.BatchNorm1d(self.embedding_dim)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)

    # @torchsnooper.snoop()
    # @profile
    def forward(self, *input):
        struct_topology = self.struct_topology
        user = input[0]
        poi = input[1]
        length = input[2]
        topology = torch.from_numpy(struct_topology).type(torch.LongTensor).cuda()
        node_feature = torch.from_numpy(np.array(range(self.poi_size))).cuda()
        node_feature = self.embed(node_feature)
        h = self.conv3(node_feature, topology)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)

        after_graph_embed = nn.Embedding(self.poi_size, self.hidden_dim, padding_idx=0).cuda()
        after_graph_embed.weight.data.copy_(h)

        user_history = self.bat_nor_embed(after_graph_embed(user).transpose(1, 2).contiguous()).transpose(1, 2)
        region_preference = self.bat_nor_embed(after_graph_embed(poi).transpose(1, 2).contiguous()).transpose(1, 2)
        poi_preference = self.bat_nor_embed(self.embed(poi).transpose(1, 2).contiguous()).transpose(1, 2)

        user_history = torch.nn.utils.rnn.pack_padded_sequence(user_history, length.cpu().tolist(), batch_first=True, enforce_sorted=False)
        user_preference, (h_n, c_n) = self.lstm(user_history) 
        user_preference = h_n.transpose(0, 1).contiguous()

        user_region_preference = torch.cat([user_preference, region_preference], -1)
        user_region_preference = torch.squeeze(user_region_preference, 1)
        user_region_preference = self.fc1(user_region_preference)

        user_region_preference = torch.unsqueeze(user_region_preference, 1)
        user_poi_preference = torch.cat([user_region_preference, poi_preference], -1)
        user_poi_preference = torch.squeeze(user_poi_preference, 1)
        output = self.fc2(user_poi_preference)

        output = F.log_softmax(output, dim=1)

        return output

    def fit(self, save_path):
        if save_path and not os.path.exists(save_path):
            f = open(save_path, "w")
            f.close()

        print("Creating save path...")
        model = self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)
        loss_function = nn.NLLLoss()
        
        loss_score = 10
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            timestamp = 0

            for train_data in process.yield_train_data(Parameters.train_review_path, self.user_history_data_dict, Parameters.MAX_LEN, shop_index, 500 * Parameters.BATCH_SIZE):
                user = train_data['user_index']
                poi = train_data['poi_index']
                length = train_data['length_index']
                label = train_data['label']

                user = np.array(user, dtype=np.int32)
                poi = np.array(poi, dtype=np.int32)
                length = np.array(length, dtype=np.int32)
                label = np.array(label, dtype=np.float32)

                batch_iter = user.shape[0] // self.batch_size
                batch_begin_time = time()
                for i in range(batch_iter + 1):
                    offset = i * self.batch_size
                    end = min(offset + self.batch_size, user.shape[0])
                    if offset == end:
                        break

                    batch_user = Variable(torch.LongTensor(user[offset:end]))
                    batch_poi = Variable(torch.LongTensor(poi[offset:end]))
                    batch_length = Variable(torch.LongTensor(length[offset:end]))
                    batch_label = Variable(torch.LongTensor(label[offset:end]))

                    if self.use_cuda:
                        batch_user, batch_poi, batch_length, batch_label = batch_user.cuda(), batch_poi.cuda(), batch_length.cuda(), batch_label.cuda()
                    optimizer.zero_grad()
                    outputs = model(batch_user, batch_poi, batch_length)
                    loss = loss_function(outputs, batch_label)
                    loss.backward()
                    optimizer.step()
                    for p in model.parameters():
                        if p.grad is not None:
                            del p.grad

                    torch.cuda.empty_cache()

                    total_loss += loss.item()
                    if self.verbose:
                        if i % 2 == 1:
                            timestamp += 1

                            print('epoch %d batch %d loss: %.6f time: %.1f s' %
                                  (epoch + 1, timestamp, total_loss / 100.0, time() - batch_begin_time))
                            if save_path and total_loss <= loss_score:
                                loss_score = total_loss
                                torch.save(self, save_path)
                                print("*******Save model successful********")

                            for p in model.parameters():
                                if p.grad is not None:
                                    del p.grad
                            torch.cuda.empty_cache()
                            total_loss = 0.0
                            batch_begin_time = time()
                            model = self.train()

                for p in model.parameters():
                    if p.grad is not None:
                            del p.grad
                torch.cuda.empty_cache()
                model = self.train()

shop_location_dict = process.create_POI_index(Parameters.shopMes_path, Parameters.shop_index_path)
shop_index = process.load_POI_index(Parameters.shop_index_path)
poincareModel = PoincareModel.load(Parameters.hieraechical_path)
type_embedding_matrix = process.create_type_matrix(Parameters.shopMes_path, poincareModel, shop_index)
user_history_data_dict = process.split_review_data(Parameters.review_path, shop_index)
struct_matrix = np.load(Parameters.concat_graph_matrix_path)

net = Net(poi_size=len(shop_index), embedding_matrix=type_embedding_matrix, struct_topology=struct_matrix, user_history_data_dict=user_history_data_dict).cuda()
net.fit("model.pkl")

count = 0
hit_recommend = 0
for test_dict in process.get_test_data(Parameters.test_review_path, user_history_data_dict, Parameters.MAX_LEN, shop_index, shop_location_dict):
    count += 1
    recommend_poi_score = dict()

    user = test_dict['user_index']
    poi = test_dict['poi_index']
    location = test_dict['location_index']

    user_history = net.embed(torch.from_numpy(np.array(user)).cuda())
    user_preference, (h_n, c_n) = net.lstm(user_history)
    user_preference = h_n.squeeze()

    k_hop_neighbor = process.construct_test_graph(Parameters.shopMes_path, shop_index, struct_matrix, location[0][0])
    for POI_id, hop_length in k_hop_neighbor.items():
        alpha = 0.9
        if hop_length != 0: 
            poi_preference = net.embed(torch. from_numpy(np.array([POI_id])).cuda())
            poi_preference = poi_preference.squeeze()
            distance_f = alpha ** (hop_length - 1)

            t = torch.dot(user_preference, poi_preference).cpu().detach().numpy()
            score = distance_f * t
            recommend_poi_score[POI_id] = score

    if len(recommend_poi_score) > 1:
        sorted_POI = sorted(recommend_poi_score.items(), key=lambda item: item[1], reverse=True)
        recommend_poi_list = []
        for item in sorted_POI[:Parameters.k]:
            recommend_poi_list.append(item[0])

        ground_truth_POI = poi[0][0]
        if ground_truth_POI in recommend_poi_list:
            hit_recommend += 1
            print(str(count) + '************' + str(hit_recommend/count))
        else:
            print(str(count) + '************' + str(hit_recommend/count))