#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load package
import os
import time
import datetime
import calendar
import json
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import joblib
import pickle
# load sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR
from sklearn_rvm import EMRVC
from sklearn_rvm import EMRVR
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb
import lightgbm as lgb
## 在線使用設置##############
import plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import plotly.express as px
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[2]:


# Metrics
def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#mre=nmape
def MRE(y_true, y_pred, capacity):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred))/capacity) * 100

def nMAE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))/y_true.mean() * 100

def MAE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def RMSE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(((y_pred - y_true) ** 2).mean())

def nRMSE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(((y_pred - y_true) ** 2).mean())/y_true.mean()*100

def cRMSE(y_true, y_pred, capacity):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(((y_pred - y_true) ** 2).mean())/capacity*100


# In[3]:


def model_build(train_x, train_y, train_idx, test_x, test_y, test_idx, model_name):
    #模型訓練
    if model_name == 'xgb':
        model = xgb.XGBRegressor(objective='reg:squarederror',
                        learning_rate=0.01, 
                        max_depth=1,
                        colsample_bytree=0.1,
                        reg_lambda=0.01,
                        seed=1,
                        subsample=0.1,
                        min_child_weight=1,
                        n_estimators=4000).fit(train_x, train_y)
    elif model_name == 'lgb':
        model = lgb.LGBMRegressor(
               boosting_type='gbdt',
                     verbose = 0,
                     learning_rate = 0.01,
                     num_leaves = 35,
                     feature_fraction=0.8,
                     bagging_fraction= 0.9,
                     bagging_freq= 8,
                     lambda_l1= 0.6,
                     lambda_l2= 0).fit(train_x, train_y)
    elif model_name == 'svr':
        model = SVR(C=1, kernel="rbf", gamma='auto').fit(train_x, train_y)
    elif model_name == 'rvm':
        model = EMRVR(kernel="rbf", gamma='auto')
        model.fit(train_x, train_y)
        joblib.dump(model,'./model/1D_power/6-17/rvm_CWB_pred_d_2021~2022.pkl')
    elif model_name == 'persistence':
#         test_y_1 = test_y[47:]
#         test_y_1 = test_y_1.reshape(-1)
#         pd.DataFrame(test_y_1).to_csv(f"15分鐘/test_y_1.csv", index=False) 
#         true_y = test_y[:-47]
#         true_y = true_y.reshape(-1)
#         pd.DataFrame(true_y).to_csv(f"15分鐘/test_y.csv", index=False) 
#         test_idx['pred'] = test_y_1
#         test_idx['true'] = true_y
        test_x = scaler_x.inverse_transform(test_x)
        test_x = test_x.reshape(-1)
        test_y = test_y.reshape(-1)
        test_idx['pred'] = test_x
        test_idx['true'] = test_y
        return test_idx



#     other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
#     'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
#     model = xgb.XGBRegressor(**other_params).fit(train_x, train_y)

# 預測
    pred_y = model.predict(test_x)
    
    
# 反正規劃
    pred_y = pred_y.reshape(-1,1)
    pred_y = scaler_y.inverse_transform(pred_y)
    pred_y = pred_y.reshape(-1)
    test_idx['pred'] = pred_y
    test_idx['true'] = test_y
    pd.DataFrame(test_idx).to_csv(f"15分鐘/test_idx.csv", index=False) 
    return test_idx


# In[4]:


merge_raw = pd.read_csv(f'Dataset/solar_汙水廠_newbig_sort(history_15m).csv')
data = merge_raw.copy()
data = data.drop_duplicates(subset=['TIME_TO_INTERVAL'], keep='last')
data = data.dropna(subset=['Power'])
data['TIME_TO_INTERVAL'] = pd.to_datetime(data['TIME_TO_INTERVAL'])
test_split_date2 = '2022-10-31'
test_split_date2 = pd.to_datetime(test_split_date2)
mask_2 = (data['TIME_TO_INTERVAL']<=test_split_date2)
data = data[mask_2]

#取4/2之後
test_split_date2 = '2022-04-02'
test_split_date2 = pd.to_datetime(test_split_date2)
mask_2 = (data['TIME_TO_INTERVAL']>=test_split_date2)
data = data[mask_2]

data = data.rename(columns={'Radiation(SDv3)(IBM)':'Radiation(SDv3)(TWC)',
                                  'WeatherType(IBM)':'WeatherType(TWC)',
                                  'WeatherType(pred)(IBM)':'WeatherType(pred)(TWC)',
                                     'Radiation(MSM)':'Radiation(SDv3)(MSM)'})

data['Radiation(SDv3)(MSM)'] = data['Radiation(SDv3)(MSM)']
data['WeatherType(MSM)'] = '無' 
data['WeatherType(pred)(MSM)'] = '無'                       
data['Hour'] = pd.to_datetime(data['TIME_TO_INTERVAL']).dt.hour
data['Date'] = pd.to_datetime(data['TIME_TO_INTERVAL']).dt.date
#因線性差值會有負數，所以將負數都已0取代
data['Radiation(SDv3)(CWB)'] = data['Radiation(SDv3)(CWB)'].where(data['Radiation(SDv3)(CWB)'] >= 0, 0)
data['Radiation(SDv3)(TWC)'] = data['Radiation(SDv3)(TWC)'].where(data['Radiation(SDv3)(TWC)'] >= 0, 0)
data['Radiation(SDv3)(OWM)'] = data['Radiation(SDv3)(OWM)'].where(data['Radiation(SDv3)(OWM)'] >= 0, 0)
data['Radiation(SDv3)(MSM)'] = data['Radiation(SDv3)(MSM)'].where(data['Radiation(SDv3)(MSM)'] >= 0, 0)
data['ClearSkyRadiation'] = data['ClearSkyRadiation'].where(data['ClearSkyRadiation'] >= 0, 0)
data['Radiation'] = data['Radiation'].where(data['Radiation'] >= 0, 0)
mask = ((data['Hour']>=5) & (data['Hour']<=18))
data = data[mask]
data = data[['TIME_TO_INTERVAL','Hour','Date','Power','Radiation','ClearSkyRadiation','Radiation(SDv3)(CWB)','Radiation(SDv3)(TWC)','Radiation(SDv3)(OWM)','Radiation(SDv3)(MSM)','minute']].reset_index(drop=True)
data = data.dropna()
data
data.to_csv(f"15分鐘/data.csv", index=False) 


# In[5]:


#輸入參數設置
def set_inputs_1(target_day_time):
    if (len(target_day_time)==3):
        hourly_attribute = np.concatenate((
                                          target_day_time['Radiation(SDv3)(CWB)'].values,
                                         ))
#         print(hourly_attribute) 
    else:
        hourly_attribute=[np.nan,np.nan,np.nan]
    inputs = hourly_attribute
    return inputs


# In[7]:


# feature_data = ['Radiation(SDv3)(CWB)','Radiation(SDv3)(TWC)','Radiation(SDv3)(OWM)','Radiation(SDv3)(MSM)']

X = []
Y = []

for i in range(len(data)):
# for i in range(100):
    target_day = data.loc[i:i].reset_index(drop=True)
    hour = target_day['Hour'].values[0]
    time = target_day['minute'].values[0]
    hour = int(hour)
    time = int(time)
    hour_y = int(hour)

#----------------test改成取前15分(feature為前15 當下 後15)
    if(i-1<0):
        target_day_time = data.loc[i:i+1].reset_index(drop=True)
        
    elif(i+1==len(data)):
        target_day_time = data.loc[i-1:i].reset_index(drop=True)
    else:
        target_day_time = data.loc[i-1:i+1].reset_index(drop=True)
#----------------test改成取前15分(feature為前15 當下 後15)

#----------------test改成取前15分(feature為前1天 當下 後1天)
#     if(i-46>0):
#         target_day_time = data.loc[i-47:i+60].reset_index(drop=True)
#         target_day_time = target_day_time[target_day_time.Hour == hour]
#         target_day_time = target_day_time[target_day_time.minute == time]
#----------------test改成取前15分(feature為前1天 當下 後1天)
    
    inputs = set_inputs_1(target_day_time)

    X.append(inputs)
    #MSM拿當日POWER，CWB、IBM、OWM拿明日POWER
    target_day_y = target_day['Date']+datetime.timedelta(days=1)
#     print("target_day_y111: ",target_day_y)
    target_day_y = data[data['Date'].isin(target_day_y)]
#     print("target_day_y222: ",target_day_y)
    target_day_y = target_day_y[target_day_y['minute'].isin([time])].reset_index(drop=True)
    target_day_y = target_day_y[target_day_y['Hour'].isin([hour_y])].reset_index(drop=True)
#     print("target_day_y333: ",target_day_y)

    if(len(target_day_y)>0):
        Y.append(target_day_y['Power'][0])
    else:
        Y.append(np.nan)
X = pd.DataFrame(X,index=None,columns=['Radiation(SDv3)(CWB)-1','Radiation(SDv3)(CWB)','Radiation(SDv3)(CWB)+1'])

data['TIME_TO_INTERVAL'] = pd.to_datetime(data['TIME_TO_INTERVAL'])
X['TIME_TO_INTERVAL'] = data['TIME_TO_INTERVAL']
Y = pd.DataFrame(Y,index=None,columns=['Power'])
Y['TIME_TO_INTERVAL'] = data['TIME_TO_INTERVAL']+datetime.timedelta(days=1)

train = pd.merge(X,Y,on = ['TIME_TO_INTERVAL'],how="inner")   
train = train.drop_duplicates(subset=['TIME_TO_INTERVAL'], keep='last')
train = train.dropna()
train


test_split_date = '2022-09-30'
test_split_date2 = '2022-10-31'
test_split_date = pd.to_datetime(test_split_date)
test_split_date2 = pd.to_datetime(test_split_date2)
train['TIME_TO_INTERVAL'] = pd.to_datetime(train['TIME_TO_INTERVAL'])
train['Date'] = train['TIME_TO_INTERVAL'].dt.date
mask_1 = (train['Date'] <= test_split_date)
train_data = train[mask_1].reset_index(drop=True)
# print(len(train_data))
train_data.to_csv(f"15分鐘/train_data.csv", index=False) 

mask_2 = (train['Date'] <= test_split_date2)
test_data = train[~mask_1&mask_2].reset_index(drop=True)
# print(len(test_data))
test_data.to_csv(f"15分鐘/test_data.csv", index=False) 
feature_data = ['Radiation(SDv3)(CWB)-1','Radiation(SDv3)(CWB)','Radiation(SDv3)(CWB)+1']
train_x = train_data[feature_data]
train_y = train_data[['Power']]
test_x = test_data[feature_data]
test_y = test_data[['Power']]


scaler_x = MinMaxScaler()
scaler_x.fit(train_x[feature_data])
train_x = scaler_x.transform(train_x[feature_data])
test_x = scaler_x.transform(test_x[feature_data])
scaler_y = MinMaxScaler()
scaler_y.fit(train_y[['Power']])
train_y = scaler_y.transform(train_y[['Power']])

train_x, train_y = np.array(train_x), np.array(train_y)
test_x, test_y = np.array(test_x), np.array(test_y)


train_idx, test_idx = pd.DataFrame(), pd.DataFrame()  
pred = model_build(train_x, train_y, train_idx, test_x, test_y, test_idx, 'rvm')
pred['pred'] = pred['pred'].where(pred['pred'] >= 0, 0)


# In[8]:


Baoshan = pd.read_csv(f'Plant_Info_Baoshan.csv', low_memory=False)
solar_capacity = Baoshan['Capacity'][1]
solar_capacity


# In[9]:


print(round(MRE(pred['true'], pred['pred'],solar_capacity),2))
print(round(nRMSE(pred['true'], pred['pred']),2))
print(round(nMAE(pred['true'], pred['pred']),2))
print(round(RMSE(pred['true'], pred['pred']),2))
print(round(MAE(pred['true'], pred['pred']),2))


# In[ ]:





# In[ ]:





# In[ ]:




