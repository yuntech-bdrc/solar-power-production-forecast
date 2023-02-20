#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
import time
import joblib
import os
from datetime import date
#資料庫
from influxdb import InfluxDBClient
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR
from sklearn_rvm import EMRVC
from sklearn_rvm import EMRVR
import xgboost as xgb
import lightgbm as lgb
#載入模型
import joblib
#繪圖工具
import plotly.graph_objects as go
import matplotlib.dates as md


# In[2]:


def split_data(data,target_day):
    power_list=['pre_Power_1','pre_Power_2','pre_Power_3']
    data_merge = data.copy()
    row = target_day.copy()
    data_power = pd.DataFrame()
    data_2 = pd.DataFrame()
    for h in range(0,3):
        data_power = data_merge[data_merge['Date'].isin(row['Date'])]
        hour_power = row['Hour']-(h+1)
        data_power = data_power[data_power['Hour'].isin(hour_power)].reset_index(drop=True)
        
        if(len(data_power)==0):
            data_2[power_list[h]] = [0]
        else:
            if(pd.isnull(data_power['Power'].values[0])):
                data_2[power_list[h]] = [0]
            else:
                data_2[power_list[h]] = data_power['Power']
    return data_2


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
#         joblib.dump(model,'./model/1H_power/cloud/rvm_pred_h_3.pkl')
    elif model_name == 'persistence':
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
    return test_idx


# In[4]:


# Metrics
def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

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


# In[5]:


merge_raw = pd.read_csv(f'solar_汙水廠(history).csv')
data = merge_raw.copy()
data = data.drop_duplicates(subset=['TIME_TO_INTERVAL'], keep='last')
data['TIME_TO_INTERVAL'] = pd.to_datetime(data['TIME_TO_INTERVAL'])
data['Hour'] = data['TIME_TO_INTERVAL'].dt.hour
data['Date'] = data['TIME_TO_INTERVAL'].dt.date
data['Date'] = pd.to_datetime(data['Date'])
#時間篩選
mask = ((data['Hour']>=6) & (data['Hour']<=17))
data = data[mask]
data = data.dropna(subset=['Power'], axis = 0, how ='any') 
data.reset_index(drop=True,inplace=True)
data = data[['TIME_TO_INTERVAL','Date','Hour','Power','Radiation','ClearSkyRadiation','Radiation(SDv3)(CWB)',
             'Radiation(SDv3)(IBM)','Radiation(SDv3)(OWM)','Radiation(MSM)','Radiation(today)(CWB)',
             'Radiation(today)(IBM)','Radiation(today)(OWM)']]

pre_datas = pd.DataFrame()
for i in range(len(data)):
    target_day = data.loc[i:i].reset_index(drop=True)
    pre_data = split_data(data,target_day)
    pre_datas = pd.concat([pre_datas,pre_data],axis=0)
# pre_datas = pre_datas.fillna(0)    
pre_datas.reset_index(drop=True,inplace=True)


# In[6]:


data['TIME_TO_INTERVAL'] = pd.to_datetime(data['TIME_TO_INTERVAL'])
mask = (data['TIME_TO_INTERVAL'] <= pd.to_datetime('2022-10-31'))
data = data[mask]
data.dropna()
test_split_date1 = '2022-02-01'
test_split_date2 = '2022-10-01'
test_split_date3 = '2022-10-31'
data = data.merge(pre_datas, how='left', left_index=True, right_index=True)


# In[7]:


mask = (data['TIME_TO_INTERVAL'] >= test_split_date1)
mask2 = (data['TIME_TO_INTERVAL'] <= test_split_date3)
data = data[mask&mask2].reset_index(drop=True)  


# In[8]:


data


# In[9]:


cloud_data = pd.read_csv(f'cloud_datas_hour.csv')
cloud_data['TIME_TO_INTERVAL'] = pd.to_datetime(cloud_data['TIME_TO_INTERVAL'])
cloud_data.rename(columns={'TIME_TO_INTERVAL': 'Date', 'time': 'Hour'}, inplace=True)
cloud_data['cloud'] = cloud_data['cloud'].where(cloud_data['cloud'] >= 0, 0)
cloud_data['low'] = cloud_data['low'].where(cloud_data['low'] >= 0, 0)
cloud_data['mid'] = cloud_data['mid'].where(cloud_data['mid'] >= 0, 0)
cloud_data['hig'] = cloud_data['hig'].where(cloud_data['hig'] >= 0, 0)
data = pd.merge(data,cloud_data,on=['Date','Hour'],how='inner')
data


# In[10]:


feature_data=[]
# for i in range(len(feature)):
for i in range(1):
    feature_data = ['pre_Power_1','pre_Power_2','pre_Power_3']
    print(feature_data)
    print(len(data))
    mask = data['TIME_TO_INTERVAL']>=test_split_date1
    mask2 = data['TIME_TO_INTERVAL']<=test_split_date2
    mask3 = data['TIME_TO_INTERVAL']<=test_split_date3
    train_data = data[mask&mask2].reset_index(drop=True)
    print(len(train_data))
    test_data = data[(~mask2)&(mask3)].reset_index(drop=True)
    print(len(test_data))
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


# In[11]:


Baoshan = pd.read_csv(f'Plant_Info_Baoshan.csv', low_memory=False)
solar_capacity = Baoshan['Capacity'][1]
solar_capacity


# In[ ]:





# In[12]:


print(round(MRE(pred['true'], pred['pred'],solar_capacity),2))
print(round(nRMSE(pred['true'], pred['pred']),2))
print(round(nMAE(pred['true'], pred['pred']),2))
print(round(RMSE(pred['true'], pred['pred']),2))
print(round(MAE(pred['true'], pred['pred']),2))


# In[ ]:





# In[13]:


line_color = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]


xtick = int(len(test_data['TIME_TO_INTERVAL'])/24)

fig_line = go.Figure()

fig_line.add_trace(go.Scatter(y = pred['true'], x=test_data['TIME_TO_INTERVAL'],
                    mode='lines',
                    name='真實值',
                    line={'dash': 'dash'},
                    line_color= '#1f77b4'))
fig_line.add_trace(go.Scatter(y = pred['pred'], x=test_data['TIME_TO_INTERVAL'],
                    mode='lines',
                    name='預測值',
                    line_color= '#ff7f0e'))
fig_line.update_layout(
    yaxis_title='發電量',
    xaxis_title='日期',
    title='彰師大汙水廠預測結果',
    font=dict(
        size=18,
    ),
#     yaxis2=dict(anchor='x', overlaying='y', side='right')
    height=450, 
    width=1500,

)

fig_line.update_xaxes(nticks=xtick)


#     fig_line.write_html(f'{folder_path}/img/{methods}_{i}.html')

fig_line.show()


# In[ ]:




