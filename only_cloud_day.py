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


# linear interpolation 線性插值
from scipy.interpolate import interp1d
def interpolate(x, kind='linear'):
    not_nan = np.logical_not(np.isnan(x))
    indices = np.arange(len(x))
#     interp = interp1d(indices[not_nan], x[not_nan], kind=kind)
    interp = interp1d(indices[not_nan], x[not_nan], kind=kind,fill_value="extrapolate")
    return interp(indices)


# In[3]:


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


# In[4]:


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
#         joblib.dump(model,'./model/1D_power/6-17/2022_rvm_CWB_pred_d.pkl')
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


# In[5]:


merge_raw = pd.read_csv(f'solar_汙水廠(history).csv', low_memory=False)
data = merge_raw.copy()
data = data.drop_duplicates(subset=['TIME_TO_INTERVAL'], keep='last')
data = data.dropna(subset=['Power'])
data['TIME_TO_INTERVAL'] = pd.to_datetime(data['TIME_TO_INTERVAL'])
#抓取>2022年的資料
test_split_date1 = '2022-02-01'
test_split_date2 = '2022-10-31'
test_split_date1 = pd.to_datetime(test_split_date1)
mask_1 = (data['TIME_TO_INTERVAL']>=test_split_date1)
mask_2 = (data['TIME_TO_INTERVAL']<=test_split_date2)
data = data[(mask_1&mask_2)]
data['Hour'] = pd.to_datetime(data['TIME_TO_INTERVAL']).dt.hour
data['Date'] = pd.to_datetime(data['TIME_TO_INTERVAL']).dt.date
data['Date'] = pd.to_datetime(data['Date'])
mask = ((data['Hour']>=6) & (data['Hour']<=17))
data = data[mask]
data = data[['TIME_TO_INTERVAL','Date','Hour','Power']].reset_index(drop=True)
data


# In[6]:


cloud_data = pd.read_csv(f'cloud_datas_hour.csv')
cloud_data['TIME_TO_INTERVAL'] = pd.to_datetime(cloud_data['TIME_TO_INTERVAL'])
cloud_data.rename(columns={'TIME_TO_INTERVAL': 'Date', 'time': 'Hour'}, inplace=True)
cloud_data['cloud'] = cloud_data['cloud'].where(cloud_data['cloud'] >= 0, 0)
cloud_data['low'] = cloud_data['low'].where(cloud_data['low'] >= 0, 0)
cloud_data['mid'] = cloud_data['mid'].where(cloud_data['mid'] >= 0, 0)
cloud_data['hig'] = cloud_data['hig'].where(cloud_data['hig'] >= 0, 0)
data = pd.merge(data,cloud_data,on=['Date','Hour'],how='inner')
data


# In[7]:


def Similarity(row_data,data):
    #獲得先前時間
    mask = (data['Date']<row_data['Date'][0])
    pre_data = data[mask]
    #獲得相同時間段
    mask2 = (pre_data['Hour'] == row_data['Hour'][0])
    pre_data_time = pre_data[mask2].reset_index(drop=True)
    #一筆一筆跟該資料做比較
    future = pd.DataFrame()
    similary = pd.DataFrame()
    for i in range(len(pre_data_time)):
        pre_row = pre_data_time.loc[i:i].reset_index(drop=True)
#         sim = distance.euclidean([pre_row['cloud']],[row_data['cloud']])
#         sim = 1 - spatial.distance.cosine([pre_row['cloud']],[row_data['cloud']])
        sim = cosine_similarity([pre_row['cloud']],[row_data['cloud']])
        print(sim)
        if(sim == 1):
#             print(pre_row['cloud'])
#             print(row_data['cloud'])
            similary = pd.concat([similary,pre_row['Power']],axis=0)
    future['cloud_Power'] = similary.mean()
    future['TIME_TO_INTERVAL'] = row_data['TIME_TO_INTERVAL']
    return future


# In[8]:


def pre_power(row_data,data):
    pre_power = pd.DataFrame()
    #獲得先前時間
    mask = (data['Date']==row_data['Date'][0]-datetime.timedelta(days=1))
    pre_data = data[mask]
    #獲得相同時間段
    mask2 = (pre_data['Hour'] == row_data['Hour'][0])
    pre_data_time = pre_data[mask2].reset_index(drop=True)
    pre_power['pre_Power'] = pre_data_time['Power']
    pre_power['TIME_TO_INTERVAL'] = row_data['TIME_TO_INTERVAL']
    return pre_power


# In[9]:


from scipy.spatial import distance
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
future_power = pd.DataFrame()
for i in tqdm(range(len(data))):
    row_data = data.loc[i:i].reset_index(drop=True)
    Similarity_data = Similarity(row_data,data)
    pre_power_data = pre_power(row_data,data)
    merge_data = pd.merge(Similarity_data,pre_power_data,on=['TIME_TO_INTERVAL'],how='inner')
    future_power = pd.concat([future_power,merge_data],axis=0)
future_power.reset_index(drop=True)


# In[10]:


future_power.head(50)


# In[11]:


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


xtick = int(len(future_power['TIME_TO_INTERVAL'])/240)

fig_line = go.Figure()

fig_line.add_trace(go.Scatter(y = future_power['cloud_Power'], x=future_power['TIME_TO_INTERVAL'],
                    mode='lines',
                    name='真實值',
                    line={'dash': 'dash'},
                    line_color= '#1f77b4'))
fig_line.add_trace(go.Scatter(y = data['Power'], x=data['TIME_TO_INTERVAL'],
                    mode='lines',
                    name='真實值',
                    line={'dash': 'dash'},
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


# In[12]:


future_power = future_power.fillna(0)
future_power.head(50)


# In[13]:


train = pd.merge(data,future_power,on = ['TIME_TO_INTERVAL'],how="inner")   


# In[14]:


train


# In[15]:


test_split_date = '2022-10-01'
test_split_date2 = '2022-10-31'
test_split_date = pd.to_datetime(test_split_date)
test_split_date2 = pd.to_datetime(test_split_date2)
train['TIME_TO_INTERVAL'] = pd.to_datetime(train['TIME_TO_INTERVAL'])
train['Date'] = train['TIME_TO_INTERVAL'].dt.date
mask_1 = (train['Date'] <= test_split_date)
train_data = train[mask_1].reset_index(drop=True)
print(len(train_data))
mask_2 = (train['Date'] <= test_split_date2)
test_data = train[~mask_1&mask_2].reset_index(drop=True)
print(len(test_data))
feature_data = ['pre_Power','cloud_Power']
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


# In[16]:


Baoshan = pd.read_csv(f'Plant_Info_Baoshan.csv', low_memory=False)
solar_capacity = Baoshan['Capacity'][1]
solar_capacity


# In[17]:


print(round(MRE(pred['true'], pred['pred'],solar_capacity),2))
print(round(nRMSE(pred['true'], pred['pred']),2))
print(round(nMAE(pred['true'], pred['pred']),2))
print(round(RMSE(pred['true'], pred['pred']),2))
print(round(MAE(pred['true'], pred['pred']),2))


# In[18]:


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





# In[ ]:




