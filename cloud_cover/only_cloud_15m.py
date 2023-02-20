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
from tqdm import tqdm


# In[2]:


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


# In[3]:


def split_data(data,target_day):
    power_list=['pre_Power_15','pre_Power_30','pre_Power_45']
    data_merge = data.copy()
    row = target_day.copy()
    #建立三個表
    data_power = pd.DataFrame()
    data_2 = pd.DataFrame()  
    #抓取該筆資料的日期和時間
    data_power = data_merge[data_merge['date'].isin(row['date'])]
    row_time = pd.to_datetime(row['TIME_TO_INTERVAL'].values[0])
    #獲得該時間的前15,30,45分鐘
    pre_time = [row_time-datetime.timedelta(minutes=15),
                row_time-datetime.timedelta(minutes=30),
                row_time-datetime.timedelta(minutes=45)]
    for i in range(len(pre_time)):
        pre_time[i] = pre_time[i].strftime("%Y-%m-%d %H:%M:%S")
       

    #依據pre_time和pre_Radiation的時間獲得該時段的power和Radiation
    #兩天都有資料的話，會有6筆，但只取前三筆(當日)
    for h in range(0,3): 
        data_power_2 = data_power[data_power['TIME_TO_INTERVAL'].isin([pre_time[h]])].reset_index(drop=True)  

        if(len(data_power_2)==0):
            data_2[power_list[h]] = [0]
#             print('---------------',data_2[power_list[h]])
        else:
            data_2[power_list[h]] = data_power_2['Power']
            
    return data_2


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
        model = EMRVR(kernel="rbf", gamma='auto',verbose=True)
        model.fit(train_x, train_y)
        #joblib.dump(model,'./model/15_minute/6-17/2022_rvm_CWB_pred_15m_low.pkl')
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


merge_data = pd.read_csv(f'./power_data/merge_weather_power_for_train15(cwb).csv', low_memory=False)
merge_data['month'] = pd.to_datetime(merge_data ['TIME_TO_INTERVAL']).dt.month
#刪除欄位中有0的值
merge_data = merge_data.drop( index = merge_data['pre_Power_15'][merge_data['pre_Power_15'] == 0].index )
merge_data = merge_data.drop( index = merge_data['pre_Power_30'][merge_data['pre_Power_30'] == 0].index )
merge_data = merge_data.drop( index = merge_data['pre_Power_45'][merge_data['pre_Power_45'] == 0].index )
merge_data = merge_data.drop( index = merge_data['pre_Radiation_15'][merge_data['pre_Radiation_15'] == 0].index )
merge_data = merge_data.drop( index = merge_data['Radiation_0'][merge_data['Radiation_0'] == 0].index )
merge_data = merge_data.drop( index = merge_data['next_Radiation_15'][merge_data['next_Radiation_15'] == 0].index )
feature = ['pre_Power_30','pre_Power_45','pre_Radiation_15','Radiation_0','next_Radiation_15','month']
mask = ((merge_data['hour']>=6) & (merge_data['hour']<=17))
merge_data = merge_data[mask].reset_index(drop=True)
mask2 = ((merge_data['TIME_TO_INTERVAL']<='2022-10-31'))
merge_data  = merge_data[mask2]
merge_data


# # 加入雲量資料

# In[6]:


def cloud_sort(row):
    if(row['cloud']<0):
        row['cloud']= np.NaN
    if(row['low']<0):
        row['low']= np.NaN
    if(row['mid']<0):
        row['mid']= np.NaN
    if(row['hig']<0):
        row['hig']= np.NaN
    return row


# In[7]:


# linear interpolation 線性插值
from scipy.interpolate import interp1d
def interpolate(x, kind='linear'):
    not_nan = np.logical_not(np.isnan(x))
    indices = np.arange(len(x))
#     interp = interp1d(indices[not_nan], x[not_nan], kind=kind)
    interp = interp1d(indices[not_nan], x[not_nan], kind=kind,fill_value="extrapolate")
    return interp(indices)


# In[8]:


#轉為小時單位
def bulid_15minute_data(data_raw):
    data_raw['TIME_TO_INTERVAL'] = pd.to_datetime(data_raw['TIME_TO_INTERVAL'])
    data_raw_2 = data_raw.groupby(pd.Grouper(key="TIME_TO_INTERVAL",freq='15min', origin='start')).mean().reset_index()
    return data_raw_2


# In[9]:


def cloud_15m(row_cloud,next_cloud):
    cloud_15 = pd.DataFrame()
    if(row_cloud['minute'][0]==10):
        cloud_15 = cloud_15.append(row_cloud,ignore_index=True)
        cloud_15 = cloud_15.append({'Date':row_cloud['Date'][0],'Hour':row_cloud['Hour'][0],'minute':15,'cloud':np.NaN,'low':np.NaN,'mid':np.NaN,'hig':np.NaN},ignore_index=True)
        cloud_15 = cloud_15.append(next_cloud,ignore_index=True)
        cloud_15 = cloud_15[['Date','Hour','minute','cloud','low','mid','hig']].reset_index(drop=True)
        cloud_15['cloud'] = interpolate(cloud_15['cloud'].values)
        cloud_15['low'] = interpolate(cloud_15['low'].values)
        cloud_15['mid'] = interpolate(cloud_15['mid'].values)
        cloud_15['hig'] = interpolate(cloud_15['hig'].values)
        cloud_15 = cloud_15.loc[1:1]
    elif(row_cloud['minute'][0]==40):
        cloud_15 = cloud_15.append(row_cloud,ignore_index=True)
        cloud_15 = cloud_15.append({'Date':row_cloud['Date'][0],'Hour':row_cloud['Hour'][0],'minute':45,'cloud':np.NaN,'low':np.NaN,'mid':np.NaN,'hig':np.NaN},ignore_index=True)
        cloud_15 = cloud_15.append(next_cloud,ignore_index=True)
        cloud_15 = cloud_15[['Date','Hour','minute','cloud','low','mid','hig']].reset_index(drop=True)
        cloud_15['cloud'] = interpolate(cloud_15['cloud'].values)
        cloud_15['low'] = interpolate(cloud_15['low'].values)
        cloud_15['mid'] = interpolate(cloud_15['mid'].values)
        cloud_15['hig'] = interpolate(cloud_15['hig'].values)
        cloud_15 = cloud_15.loc[1:1]
    elif(row_cloud['minute'][0]==30 or row_cloud['minute'][0]==0):
        cloud_15 = row_cloud[['Date','Hour','minute','cloud','low','mid','hig']]
    else:
         cloud_15[['Date','Hour','minute','cloud','low','mid','hig']] = np.nan
    return cloud_15


# In[10]:


cloud = pd.read_csv(f'./cloud_cover/material/cloud_datas.csv', low_memory=False)
cloud.rename(columns={'time': 'minute'}, inplace=True)
cloud['Date'] = cloud['TIME_TO_INTERVAL']
cloud['Hour'] = cloud['minute']//60
cloud['minute'] = cloud['minute']%60
#使用線性插植
cloud = cloud.apply(lambda x: cloud_sort(x), axis=1)
cloud['cloud'] = interpolate(cloud['cloud'].values)
cloud['low'] = interpolate(cloud['low'].values)
cloud['mid'] = interpolate(cloud['mid'].values)
cloud['hig'] = interpolate(cloud['hig'].values)
cloud = cloud[['Date','Hour','minute','cloud','low','mid','hig']]
cloud_15minute = pd.DataFrame()
for i in tqdm(range(len(cloud)-1)):
    row_cloud = cloud.loc[i:i].reset_index(drop=True)
    next_cloud = cloud.loc[i+1:i+1].reset_index(drop=True)
    cloud_15 = cloud_15m(row_cloud,next_cloud)
    cloud_15minute = pd.concat([cloud_15minute,cloud_15],axis=0)
cloud_15minute.reset_index(drop=True)


# In[11]:


cloud_15minute['TIME_TO_INTERVAL'] = pd.to_datetime(cloud_15minute.apply(
        lambda row: '{} {:02d}:{:02d}:00'.format(row['Date'], int(row['Hour']), int(row['minute'])), axis=1))
cloud_15minute = cloud_15minute[['TIME_TO_INTERVAL','cloud','low','mid','hig']].reset_index(drop=True)
merge_data['TIME_TO_INTERVAL'] = pd.to_datetime(merge_data['TIME_TO_INTERVAL'])
cloud_15minute['TIME_TO_INTERVAL'] = pd.to_datetime(cloud_15minute['TIME_TO_INTERVAL'])
merge_data = pd.merge(merge_data,cloud_15minute,on = ['TIME_TO_INTERVAL'],how="inner")   


# In[12]:


merge_data


# In[13]:


Baoshan = pd.read_csv(f'Plant_Info_Baoshan.csv', low_memory=False)
solar_capacity = Baoshan['Capacity'][1]
solar_capacity


# In[14]:


# test_split_date = '2022-05-20'
# test_split_date_2 = '2022-09-15'
test_split_date = '2022-10-01'
test_split_date_2 = '2022-10-31'
anser = []
feature = ['cloud','low','mid','hig']
result = pd.DataFrame()
for i in range(len(feature)):
# for i in range(1):
    feature_data = ['pre_Power_15','pre_Power_30','next_Radiation_15']
#     feature_data =['pre_Power_15']
    feature_data.append(feature[i])
    mask = merge_data['TIME_TO_INTERVAL']<test_split_date
    mask2 = merge_data['TIME_TO_INTERVAL']>=test_split_date
    mask3 = merge_data['TIME_TO_INTERVAL']<=test_split_date_2
    train_data = merge_data[mask]
    test_data = merge_data[(mask2&mask3)]
    print('train_data:',len(train_data),'test_data:',len(test_data))
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
    pre_result=[]
    pre_result.append(round(MRE(pred['true'], pred['pred'],solar_capacity),2))
    pre_result.append(round(nRMSE(pred['true'], pred['pred']),2))
    pre_result.append(round(nMAE(pred['true'], pred['pred']),2))
    pre_result.append(round(RMSE(pred['true'], pred['pred']),2))
    pre_result.append(round(MAE(pred['true'], pred['pred']),2))
    result[feature[i]] = pre_result


# In[15]:


result


# In[16]:


print(round(MRE(pred['true'], pred['pred'],solar_capacity),2))
print(round(nRMSE(pred['true'], pred['pred']),2))
print(round(nMAE(pred['true'], pred['pred']),2))
print(round(RMSE(pred['true'], pred['pred']),2))
print(round(MAE(pred['true'], pred['pred']),2))


# In[17]:


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




