#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
def yield_train_data(batch_size):
    df_train = pd.read_csv("newData/Review_train.csv", encoding="utf-8", low_memory=False)
    user_history_dict = {}
    print("Loading train data....")
    labelIndex = []
    userIndex = []
    userHistoryIndex = []
    poiIndex = []
    
    count = 0
    for review in df_train.iterrows():
        userId = review[1]["userID"]        
        poiId = review[1]["shopID"]+1          # 0， pad
        score = int(review[1]["score"])-1
        
        if userId not in user_history_dict.keys():
            user_history_dict[userId] = [poiId]
        
        else:
            history_length = len(user_history_dict[userId])
            if history_length >= 3:           # 控制输入网络的是用户当前访问不少于3个兴趣点
                tempList = user_history_dict[userId].copy()   # 一层列表的深拷贝
                userIndex.append(userId)
                userHistoryIndex.append(tempList)
                poiIndex.append(poiId)
                labelIndex.append(score)
                
                count += 1
                if count == batch_size:
                    yield {"userIndex": userIndex, "userHistoryIndex": userHistoryIndex, "poiIndex": poiIndex, "labelIndex": labelIndex}
                    labelIndex = []
                    userIndex = []
                    userHistoryIndex = []
                    poiIndex = []
                    count = 0
            user_history_dict[userId].append(poiId)
    if count!=0:
        yield {"userIndex": userIndex, "userHistoryIndex": userHistoryIndex, "poiIndex": poiIndex, "labelIndex": labelIndex}
        
    np.save('newData/UserHistoryData.npy', user_history_dict)  

def yield_test_data(batch_size):
    userHistoryDict = np.load('newData/UserHistoryData.npy', allow_pickle=True).item()
    df_testReview = pd.read_csv("newData/Review_test.csv", encoding="utf-8", low_memory=False)

    print("Loading test data....")
    userIndex = []
    userHistoryIndex = []
    poiIndex = []
    labelIndex = []
    
    count = 0
    byGroup = df_testReview.groupby("userID")
    for index, group in byGroup:
        userId = group.iat[0,0]
        userIndex.append(userId)
        userHistoryIndex.append(userHistoryDict[userId])
        for review in group.iterrows():
            poiId = review[1]["shopID"]+1
            score = int(review[1]["score"])-1
            poiIndex.append(poiId)
            labelIndex.append(score)
        
        count += 1
        if count == batch_size:
            yield {"userIndex": userIndex, "userHistoryIndex": userHistoryIndex, "poiIndex": poiIndex, "labelIndex":labelIndex}
            userIndex = []
            userHistoryIndex = []
            poiIndex = []
            labelIndex = []
            count = 0
            
    if count!=0:
        yield {"userIndex": userIndex, "userHistoryIndex": userHistoryIndex, "poiIndex": poiIndex, "labelIndex":labelIndex}

