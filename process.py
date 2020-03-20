from utils import Parameters
import pandas as pd
import numpy as np
from pandas import DataFrame
import math
from math import cos, sin
from gensim.models.poincare import PoincareModel
import networkx as nx

pd.set_option('display.max_columns', None)  # 显示时不折叠


def process_nan(string):
    string = 0 if string == '-' else string
    return string


def create_POI_index(shopMes_path, shopIndex_path):
    print("Creating shop index...")
    df = pd.read_csv(shopMes_path, encoding="gbk")
    f = open(shopIndex_path, "w")
    for i, v in df["shopId"].items():
        f.write(str(v) + " " + str(i+1) + "\n")
    f.close()
    print("shop 总数：" + str(i + 1))
    shop_location_dict = {}
    for index, row in df.iterrows():
        shop_location_dict[row['shopId']] = (row['longitude'], row['latitude'])
    return shop_location_dict


def load_POI_index(shopIndex_path):
    print("Loading shop index...")
    shop_index = {}
    f = open(shopIndex_path, "r")
    for pair in f:
        pair = pair.strip().split()
        shopId = pair[0]
        idx = pair[1]
        shop_index[shopId] = int(idx)
    f.close()
    return shop_index


def create_type_matrix(shopMes_path, poincareModel, shop_index):
    print("Creating type matrix...")
    poi_num = len(shop_index) + 1
    type_embedding_matrix = np.zeros((poi_num, Parameters.type_embedding_dim), dtype=np.float32)
    df = pd.read_csv(shopMes_path, encoding="gbk")
    for idx, shopType in df["cuisine"].items():
        try:
            type_embedding_matrix[idx + 1] = poincareModel.kv.word_vec(shopType.strip())
        except:
            print("Did not found %s in pre-trained model" % shopType)
    return type_embedding_matrix


def create_attribute_matrix(shopMes_path):
    print("Creating attribute matrix...")
    df = pd.read_csv(shopMes_path, encoding="gbk")
    df["per_consume"] = df.apply(lambda x: process_nan(x["per_consume"]), axis=1)
    df["star"] = df.apply(lambda x: process_nan(x["star"]), axis=1)
    attribute_embedding_matrix = df[["review_count", "per_consume", "star", "taste_score", "env_score", "service_score"]].astype(np.float32).values
    attribute_embedding_matrix = np.vstack((np.zeros((1, 6), dtype=np.float32), attribute_embedding_matrix))
    return attribute_embedding_matrix


def create_embedding_matrix(type_embedding_matrix, attribute_embedding_matrix):
    print("Creating embedding matrix...")
    embedding_matrix = np.concatenate((type_embedding_matrix, attribute_embedding_matrix), axis=1)
    np.save(Parameters.concat_embedding_matrix_path, embedding_matrix)
    return embedding_matrix


def trans(shop_id, shop_index):
    if shop_id in shop_index.keys():
        shop_id = shop_index[shop_id]
    else:
        shop_id = None
    return shop_id


def split_review_data(review_path, shop_index):
    user_history_data_dict = {}

    df_train_demo = DataFrame({'user_id': [0],
                     'user_name': ['自定义'],
                     'shop_id': [0],
                     'shop_name': ['自定义'],
                     'comment_time': ['1996-07-18'],
                     'star': [0]})
    df_test_demo = DataFrame({'user_id': [0],
                               'user_name': ['自定义'],
                               'shop_id': [0],
                               'shop_name': ['自定义'],
                               'comment_time': ['1996-07-18'],
                               'star': [0]})

    df = pd.read_csv(review_path, encoding='utf-8')
    # df['shop_id'] = df.apply(lambda x: trans(x['shop_id']), axis=1)  # shop_id转index_id
    grouped = df.groupby('user_id')
    for user_id, group in grouped:
        user_count = group.shape[0]
        if user_count < Parameters.history_review_count:         # 过滤少于10个历史数据的用户
            continue
        else:
            df_train_data = group[0:int(group.shape[0]*0.8)]   # 80%作为训练集
            df_test_data = group[int(group.shape[0]*0.8):]

            df_train_demo = pd.concat([df_train_demo, df_train_data])
            df_test_demo = pd.concat([df_test_demo, df_test_data])

            for shop_id in df_train_data['shop_id']:
                if user_id not in user_history_data_dict.keys():
                    if str(shop_id) in shop_index.keys():
                        user_history_data_dict[user_id] = [shop_index[str(shop_id)]]

                else:
                    if str(shop_id) in shop_index.keys():
                        shop_id = shop_index[str(shop_id)]
                        user_history_data_dict[user_id].append(shop_id)

    f = open(Parameters.user_history_review_path, 'w')   # train部分用户历史记录dict保存
    f.write(str(user_history_data_dict))
    f.close()

    df_train_demo = df_train_demo[1:]
    df_test_demo = df_test_demo[1:]
    df_train_demo.to_csv(Parameters.train_review_path, header=True, index=False)
    df_test_demo.to_csv(Parameters.test_review_path, header=True, index=False)

    return user_history_data_dict


def rad(d):
    pi = 3.1415926
    return d * pi / 180.0


def get_distance(lat1, lng1, lat2, lng2):
    EARTH_REDIUS = 6378.137
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(math.pow(sin(a / 2), 2) + cos(radLat1) * cos(radLat2) * math.pow(sin(b / 2), 2)))
    s = s * EARTH_REDIUS
    return s


def construct_graph(shopMes_path, shop_index):
    print("Counstruct graph...")
    struct_list = []
    df = pd.read_csv(shopMes_path, encoding="gbk", low_memory=False)
    df_copy = df
    for index, row in df.iterrows():
        poi_id_1 = row['shopId']
        longitude_1 = row['longitude']
        latitude_1 = row['latitude']

        for index1, row1 in df_copy.iterrows():
            poi_id_2 = row1['shopId']
            longitude_2 = row1['longitude']
            latitude_2 = row1['latitude']

            d = get_distance(latitude_1, longitude_1, latitude_2, longitude_2) * 1000
            if poi_id_1 != poi_id_2 and d < Parameters.DISTANCE:
                struct_list.append([shop_index[str(poi_id_1)], shop_index[str(poi_id_2)]])
    struct_matrix = np.array(struct_list).T
    np.save(Parameters.concat_graph_matrix_path, struct_matrix)
    return struct_matrix


def construct_test_graph(shopMes_path, shop_index, struct_matrix, location_set):
    struct_list = struct_matrix.T.tolist()

    user_location_id = 100000   # 当前用户位置作为一个隐节点嵌入图中
    df = pd.read_csv(shopMes_path, encoding="gbk", low_memory=False)
    for index, row in df.iterrows():
        poi_id_1 = row['shopId']
        longitude_1 = row['longitude']
        latitude_1 = row['latitude']

        longitude_2 = location_set[0]
        latitude_2 = location_set[1]

        d = get_distance(latitude_1, longitude_1, latitude_2, longitude_2) * 1000
        if d < Parameters.DISTANCE:
            struct_list.append([shop_index[str(poi_id_1)], user_location_id])
            struct_list.append([user_location_id, shop_index[str(poi_id_1)]])
    node_list = [x for x in range(len(shop_index))]
    node_list.append(user_location_id)

    G = nx.DiGraph()
    G.add_nodes_from(node_list)
    G.add_edges_from(struct_list)
    hop_length = nx.single_source_shortest_path_length(G, user_location_id, 5)

    return hop_length


def yield_train_data(data_path, user_history_data_dict, max_len, shop_index, batch_size):
    print("Loading train data...")
    label = []
    user_index = []
    poi_index = []
    length_index = []

    count = 0
    f = open(data_path, "r")
    line_count = 0
    for line in f:
        if line_count == 0:
            line_count = 1
            continue
        else:
            line = line.strip().split(",")
            user = user_history_data_dict[int(line[4])]       # user历史数据list, int:list
            if len(user) > max_len:
                user = user[-max_len:]
                length_index.append(max_len)
            else:
                length_index.append(len(user))
                user = user + [0] * (max_len - len(user)) 
        
            poi = [shop_index[str(line[1])]]                  # str : str
            user_index.append(user)
            poi_index.append(poi)

            label.append(int(line[3]) - 1)

            count += 1
            if count == batch_size:
                yield {"user_index": user_index, "poi_index": poi_index, "length_index": length_index, "label": label}
                label = []
                user_index = []
                poi_index = []
                length_index = []
                count = 0
    if count != 0:
        yield {"user_index": user_index, "poi_index": poi_index, "length_index": length_index, "label": label}
    f.close()


def get_test_data(data_path, user_history_data_dict, max_len, shop_index, shop_location_dict):
    count = 0
    location_index = []
    user_index = []
    poi_index = []

    f = open(data_path, "r")
    line_count = 0
    for line in f:
        if line_count == 0:
            line_count = 1
            continue
        else:
            line = line.strip().split(",")

            user = user_history_data_dict[int(line[4])]  # user历史数据list, int:list
            # if len(user) > max_len:
            #     user = user[-max_len:]
            # else:
            #     user = user + [0] * (max_len - len(user))  
            poi = [shop_index[line[1]]]  # str : str
            location = [shop_location_dict[int(line[1])]]

            user_index.append(user)
            poi_index.append(poi)
            location_index.append(location)

            count += 1
            if count == 1:
                yield {"user_index": user_index, "poi_index": poi_index, "location_index": location_index}
                location_index = []
                user_index = []
                poi_index = []
                count = 0
    f.close()


# process_review_data("E:/review_utf-8.csv")

# dict从文件中读取
# f = open('temp.txt','r')
# a = f.read()
# dict_name = eval(a)
# f.close()
