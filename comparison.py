# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:36:48 2022

@author: Nancy Wang
"""


import pandas as pd
import os
import numpy as np
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from random import random
from sklearn import metrics
import warnings


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,LSTM,Dropout  
import keras_metrics as km


from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor as XGBR




Attack_category = {'XMRIGCC CryptoMiner', 'Bruteforce', 'Bruteforce-XML', 'Probing'} #[0,1,2,3]
Benign_category = {'Benign', 'Background'}

class ReadFlow:
    def LoadData(path):
        data = pd.DataFrame(pd.read_csv(path))
        return data
    
#按照时间序列的测试集划分
def SplitTrain(data,test_ratio):
    np.random.seed(43) #keep the split result consistent
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices =shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

def RF(x_train,y_train,x_test,y_test):
    rfc = RandomForestClassifier(n_estimators=50,max_depth=3,max_features=2,min_samples_leaf=7)
    rfc.fit(x_train,y_train)
    pred = rfc.predict(x_test)
    print(metrics.classification_report(pred,y_test))
    
def LR(x_train,y_train,x_test,y_test):
    clf = LogisticRegression()
    # 输入训练数据
    clf.fit(x_train,y_train)
    pred = clf.predict(x_test)    
    print(metrics.classification_report(pred,y_test))
    
def XGB(x_train,y_train,x_test,y_test):
    reg = XGBR(n_estimators=50).fit(x_train,y_train)
    pred = reg.predict(x_test) #传统接口predict
    pred = pred.astype(np.int64)
    print(metrics.classification_report(pred,y_test))
    
def DNN(x_train,y_train,x_test,y_test):
    # 创建网络
    model=Sequential()
    model.add(Dense(input_dim=79,units=512,activation='relu'))
    model.add(Dense(units=156,activation='relu'))
    model.add(Dense(units=1,activation='softmax'))
    
    
    # 模型编译
    model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy',km.categorical_precision(), km.binary_recall(),km.f1_score()])
    
    #model.summary()
    
    # 训练模型
    model.fit(x_train, y_train, batch_size=64,epochs=10)
    
    # 模型评估
    score = model.evaluate(x_test,y_test)
    print('Total loss on Testiong Set : ',score[0])
    print('Accuracy of Testiong Set : ',score[1])
    
    #预测结果
    result = model.predict(x_test)
    result = result.astype(np.int64)   
    
    print(metrics.classification_report(result,y_test))
    
def LSTM2(x_train,y_train,x_test,y_test):
    x_train.resize((len(x_train),79,1))  
    x_test.resize((len(x_test),79,1))

    
    # 创建网络
    model=Sequential()
    model.add(LSTM(input_dim=1,units=512,activation='relu',return_sequences=True))
    model.add(LSTM(units=156,activation='relu',return_sequences=True))
    model.add(LSTM(units=1,activation='softmax',return_sequences=True))
    model.add(Dropout(0.5))
    
    # 模型编译
    model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy',km.categorical_precision(), km.binary_recall(),km.f1_score()])
    
    #model.summary()
    
    # 训练模型
    model.fit(x_train, y_train, batch_size=32,epochs=50)
    
    # 模型评估
    score = model.evaluate(x_test,y_test)
    print('Total loss on Testiong Set : ',score[0])
    print('Accuracy of Testiong Set : ',score[1])
    

def classifier(x_train,y_train,x_test,y_test):
    print("***********************RandomForest************************")
    RF(x_train,y_train,x_test,y_test)
    print("*********************LogisticRegression**********************")
    #LR(x_train,y_train,x_test,y_test)
    print("*************************xgboost************************")
    #XGB(x_train,y_train,x_test,y_test)
    print("*************************DNN**************************")
    #DNN(x_train,y_train,x_test,y_test)
    print("*************************LSTM**************************")
    #LSTM2(x_train,y_train,x_test,y_test)





if __name__ == '__main__':
    FlowData = ReadFlow.LoadData(HIKARI_path) #'ALLFLOWMETER_HIKARI2021.csv'
    
    Benign = FlowData.loc[FlowData ['traffic_category'] == 'Benign']
    Benign.loc[Benign.traffic_category == 'Benign', 'traffic_category'] = 0
    
    
    Attack = FlowData[(FlowData.traffic_category == 'XMRIGCC CryptoMiner' ) | (FlowData.traffic_category =='Bruteforce')|   (FlowData.traffic_category == 'Bruteforce-XML' ) | (FlowData.traffic_category =='Probing')]
    Attack.loc[Attack.traffic_category == 'XMRIGCC CryptoMiner', 'traffic_category'] = 1
    Attack.loc[Attack.traffic_category == 'Bruteforce', 'traffic_category'] = 2
    Attack.loc[Attack.traffic_category == 'Bruteforce-XML', 'traffic_category'] = 3
    Attack.loc[Attack.traffic_category == 'Probing', 'traffic_category'] = 4
    
    
    # 攻击类别的正常分类。将四种类型切分成训练集和测试集
    Train_Attack, Test_Attack = SplitTrain(Attack, 0.2)
    
    for add_number in range(0,110,10):
        Train_Attack, Test_Attack = SplitTrain(Attack, 0.2)
        add_ratio = add_number/len(Test_Attack)
        x, df_Benign_add = SplitTrain(Benign,add_ratio)
        print(len(Test_Attack))
        Test_Attack = Test_Attack.append(df_Benign_add,ignore_index=True)
        print(len(Test_Attack))
        
        x_train = Train_Attack.iloc[:,np.arange(7, 86)].values
        y_train = np.array((Train_Attack.traffic_category.tolist())).reshape((-1,1))
    
        x_test = Test_Attack.iloc[:,np.arange(7, 86)].values
        y_test = np.array((Test_Attack.traffic_category.tolist())).reshape((-1,1))
        
        print("======================add_ratio is" + str(add_ratio) +"!!!!!============================")
        
        classifier(x_train,y_train,x_test,y_test)



    
