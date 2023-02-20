#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime
from influxdb import InfluxDBClient
from tqdm import tqdm
databasename = ['MG1']
client = InfluxDBClient('120.107.146.56', 8086, 'ncue01', 'Q!A@Z#WSX', 'MG1') 


# In[2]:


import os
from datetime import date
#資料庫
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
import matplotlib.dates as md
from tqdm import tqdm 


# In[3]:


## 在線使用設置##############
import plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import plotly.express as px
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[4]:


# def get_old_power():
#     path="C:\\Users\\IDSL\\Desktop\\G.Z\\太陽能\\太陽能發電\\天氣資料爬蟲與合併\\power_data\\MG1_PV"
#     filenames = os.listdir(path)
#     old_power = pd.DataFrame()
#     for excel in tqdm(filenames):
#         power=pd.read_csv(os.path.join(path,excel))
#         old_power=pd.concat([old_power,power],axis=0, ignore_index=True)
#     return old_power
# old_power = get_old_power()
# old_power = old_power.rename(columns={'time':'TIME_TO_INTERVAL'})
# old_power = old_power.sort_values(by=['TIME_TO_INTERVAL'])
# old_power = bulid_15minute_data(old_power)
# old_power.to_csv('power_data/original/save/merge_power_data(old_15min).csv',index=None)


# In[5]:


#抓取資料庫全部資料
def get_power(date):
    tablename = 'MG1_PV'
    sql = f"SELECT * FROM {tablename} where Time >= '{date}' - 8h"#中原標準時間，為中華民國現行採用的標準時間，比世界協調時間快八個小時
    print(sql)
    result = client.query(sql) 
    #result = client.query(f'SELECT * FROM {tablename}') 
    data =  list(result.get_points())
    data = pd.DataFrame(data)
    data = data.rename(columns={'Time':'TIME_TO_INTERVAL'})
    data = bulid_15minute_data(data)
    #data_2.to_csv('power_data/original/save/merge_power_data(15min).csv',index=None)
    return data


# In[6]:


#轉為小時單位
def bulid_15minute_data(data_raw):
    data_raw['TIME_TO_INTERVAL'] = pd.to_datetime(data_raw['TIME_TO_INTERVAL'])
    data_raw_2 = data_raw.groupby(pd.Grouper(key="TIME_TO_INTERVAL",freq='15min', origin='start')).mean().reset_index()
    return data_raw_2


# In[7]:


# day = datetime.datetime.today()
# day = pd.to_datetime(day, format='%Y%m%d')
# day = pd.to_datetime(str(day.year)+'-'+str(day.month)+'-'+str(day.day))

# data = get_power(day)
# data


# # 正式開始

# In[8]:


def split_data(data,target_day):
    power_list=['pre_Power_15','pre_Power_30','pre_Power_45']
    Radiation_list=['pre_Radiation_15','Radiation_0','next_Radiation_15']
    data_merge = data.copy()
    row = target_day.copy()
    #建立三個表
    data_power = pd.DataFrame()
    data_Radiation = pd.DataFrame()
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
       
    #獲得該時間的前15,現在,未來15分鐘
    pre_Radiation = [row_time-datetime.timedelta(minutes=15),
                    row_time,
                    row_time+datetime.timedelta(minutes=15)]
    for i in range(len(pre_Radiation)):
        pre_Radiation[i] = pre_Radiation[i].strftime("%Y-%m-%d %H:%M:%S")
    
    #獲得該日期的當天和明天全部資料
    row_date = pd.to_datetime(row['date'].values[0])
    next_date = [row_date,
                 row_date+datetime.timedelta(days=1)]   
    for i in range(len(next_date)):
        next_date[i] = next_date[i].strftime("%Y-%m-%d")
    data_merge['date'] = data_merge['date'].apply(lambda x: x.strftime('%Y-%m-%d'))     
    data_Radiation = data_merge[data_merge['date'].isin(next_date)]

    #依據pre_time和pre_Radiation的時間獲得該時段的power和Radiation
    #兩天都有資料的話，會有6筆，但只取前三筆(當日)
    #因如為24:00,則需要抓取當天和明天資料,並只取當天
    for h in range(0,3): 

        data_power_2 = data_power[data_power['TIME_TO_INTERVAL'].isin([pre_time[h]])].reset_index(drop=True)  
        data_Radiation_2 = data_Radiation[data_Radiation['TIME_TO_INTERVAL'].isin([pre_Radiation[h]])].reset_index(drop=True)

    
        if(len(data_power_2)==0):
            data_2[power_list[h]] = [0]
#             print('---------------',data_2[power_list[h]])
        else:
            data_2[power_list[h]] = data_power_2['Power']
            
        if(len(data_Radiation_2)==0):
            data_2[Radiation_list[h]] = [0]
        else:
            data_2[Radiation_list[h]] = data_Radiation_2['Radiation(today)(CWB)']
    return data_2


# In[9]:


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
        joblib.dump(model,'./model/15_minute/rvm_pred(cwb).pkl')
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


# In[10]:


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


# In[11]:


# linear interpolation 線性插值
from scipy.interpolate import interp1d
def interpolate(x, kind='linear'):
    not_nan = np.logical_not(np.isnan(x))
    indices = np.arange(len(x))
#     interp = interp1d(indices[not_nan], x[not_nan], kind=kind)
    interp = interp1d(indices[not_nan], x[not_nan], kind=kind,fill_value="extrapolate")
    return interp(indices)


# In[12]:


# # 依據時間將15分鐘的發電資料和天氣資料合併
# data = pd.read_csv('power_data/merge_alldata_15.csv')
# data = data.rename(columns={'kP':'Power'})
# data['hour'] = pd.to_datetime(data['TIME_TO_INTERVAL']).dt.hour
# data['date'] = pd.to_datetime(data['TIME_TO_INTERVAL']).dt.date
# data = data[['TIME_TO_INTERVAL','date','hour','Power']]
# data = data.dropna(subset=['Power']).reset_index(drop=True)
# data = data.drop_duplicates(['TIME_TO_INTERVAL'], keep="last").reset_index(drop=True)
# weatherdata = pd.read_csv('dataset/solar_汙水廠(history).csv')
# weatherdata = weatherdata.rename(columns={'Radiation(SDv3)(IBM)':'Radiation(SDv3)(TWC)',
#                                           'Radiation(MSM)':'Radiation(SDv3)(MSM)'})
# weatherdata['hour'] = pd.to_datetime(weatherdata['TIME_TO_INTERVAL']).dt.hour
# weatherdata['date'] = pd.to_datetime(weatherdata['TIME_TO_INTERVAL']).dt.date
# weatherdata = weatherdata[['date','hour','Radiation','ClearSkyRadiation','Radiation(SDv3)(CWB)',
#                            'Radiation(SDv3)(TWC)','Radiation(SDv3)(OWM)','Radiation(SDv3)(MSM)',
#                            'Radiation(today)(CWB)','Radiation(today)(IBM)','Radiation(today)(OWM)']]
# merge_data = pd.merge(data,weatherdata,on=['date','hour'],how='inner')
# merge_data['minute'] = pd.to_datetime(merge_data['TIME_TO_INTERVAL']).dt.minute
# merge_data = merge_data.drop_duplicates(['TIME_TO_INTERVAL'], keep="last").reset_index(drop=True)
# #對可能會有缺失值的欄位做線性差值
# merge_data['Radiation(SDv3)(CWB)'] = interpolate(merge_data['Radiation(SDv3)(CWB)'].values)
# merge_data['Radiation(SDv3)(TWC)'] = interpolate(merge_data['Radiation(SDv3)(TWC)'].values)
# merge_data['Radiation(SDv3)(OWM)'] = interpolate(merge_data['Radiation(SDv3)(OWM)'].values)
# merge_data['Radiation(SDv3)(MSM)'] = interpolate(merge_data['Radiation(SDv3)(MSM)'].values)
# merge_data['ClearSkyRadiation'] = interpolate(merge_data['ClearSkyRadiation'].values)
# merge_data['Radiation(SDv3)(MSM)'] = interpolate(merge_data['Radiation(SDv3)(MSM)'].values)
# #因線性差值會有負數，所以將負數都已0取代
# merge_data['Radiation(SDv3)(CWB)'] = merge_data['Radiation(SDv3)(CWB)'].where(merge_data['Radiation(SDv3)(CWB)'] >= 0, 0)
# merge_data['Radiation(SDv3)(TWC)'] = merge_data['Radiation(SDv3)(TWC)'].where(merge_data['Radiation(SDv3)(TWC)'] >= 0, 0)
# merge_data['Radiation(SDv3)(OWM)'] = merge_data['Radiation(SDv3)(OWM)'].where(merge_data['Radiation(SDv3)(OWM)'] >= 0, 0)
# merge_data['Radiation(SDv3)(MSM)'] = merge_data['Radiation(SDv3)(MSM)'].where(merge_data['Radiation(SDv3)(MSM)'] >= 0, 0)
# merge_data['ClearSkyRadiation'] = merge_data['ClearSkyRadiation'].where(merge_data['ClearSkyRadiation'] >= 0, 0)
# merge_data['Radiation'] = merge_data['Radiation'].where(merge_data['Radiation'] >= 0, 0)

# pre_datas = pd.DataFrame()
# for i in tqdm(range(len(merge_data))):
#     target_day = merge_data.loc[i:i].reset_index(drop=True)
#     pre_data = split_data(merge_data,target_day)
#     pre_datas = pd.concat([pre_datas,pre_data],axis=0)
# pre_datas = pre_datas.fillna(0)    
# pre_datas.reset_index(drop=True,inplace=True)
# pre_datas
# merge_data = merge_data.merge(pre_datas, how='left', left_index=True, right_index=True)
# merge_data.to_csv('power_data/merge_weather_power_for_train15(cwb).csv',index=None)


# In[ ]:


# merge_data = pd.read_csv('power_data/merge_weather_power_for_train15.csv', low_memory=False)
# merge_data['month'] = pd.to_datetime(merge_data ['TIME_TO_INTERVAL']).dt.month
# merge_data = merge_data.drop( index = merge_data['pre_Power_15'][merge_data['pre_Power_15'] == 0].index )
# merge_data = merge_data.drop( index = merge_data['pre_Power_30'][merge_data['pre_Power_30'] == 0].index )
# merge_data = merge_data.drop( index = merge_data['pre_Power_45'][merge_data['pre_Power_45'] == 0].index )
# merge_data = merge_data.drop( index = merge_data['pre_Radiation_15'][merge_data['pre_Radiation_15'] == 0].index )
# merge_data = merge_data.drop( index = merge_data['Radiation_0'][merge_data['Radiation_0'] == 0].index )
# merge_data = merge_data.drop( index = merge_data['next_Radiation_15'][merge_data['next_Radiation_15'] == 0].index )
# merge_data


# In[ ]:


merge_data = pd.read_csv('power_data/merge_weather_power_for_train15(twc).csv', low_memory=False)
merge_data['month'] = pd.to_datetime(merge_data ['TIME_TO_INTERVAL']).dt.month
#刪除欄位中有0的值
merge_data = merge_data.drop( index = merge_data['pre_Power_15'][merge_data['pre_Power_15'] == 0].index )
merge_data = merge_data.drop( index = merge_data['pre_Power_30'][merge_data['pre_Power_30'] == 0].index )
merge_data = merge_data.drop( index = merge_data['pre_Power_45'][merge_data['pre_Power_45'] == 0].index )
merge_data = merge_data.drop( index = merge_data['pre_Radiation_15'][merge_data['pre_Radiation_15'] == 0].index )
merge_data = merge_data.drop( index = merge_data['Radiation_0'][merge_data['Radiation_0'] == 0].index )
merge_data = merge_data.drop( index = merge_data['next_Radiation_15'][merge_data['next_Radiation_15'] == 0].index )
feature = ['pre_Power_30','pre_Power_45','pre_Radiation_15','Radiation_0','next_Radiation_15','month']
test_split_date = '2022-05-20'
test_split_date_2 = '2022-09-15'
anser = []
# for i in range(len(feature)):
for i in range(1):
    #feature_data = ['pre_Power_15','next_Radiation_15','pre_Power_30']
    feature_data =['pre_Power_15']
#     feature_data.append(feature[i])
    mask = merge_data['TIME_TO_INTERVAL']<=test_split_date
    mask2 = merge_data['TIME_TO_INTERVAL']>test_split_date
    mask3 = merge_data['TIME_TO_INTERVAL']<=test_split_date_2
    train_data = merge_data[mask]
    test_data = merge_data[(mask2&mask3)]
    #print(test_data)
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
    pred = model_build(train_x, train_y, train_idx, test_x, test_y, test_idx, 'persistence')
    anser.append(round(nRMSE(pred['true'], pred['pred']),2))
    anser.append(round(nMAE(pred['true'], pred['pred']),2))
    anser.append(round(RMSE(pred['true'], pred['pred']),2))
    anser.append(round(MAE(pred['true'], pred['pred']),2))
    print(round(nRMSE(pred['true'], pred['pred']),2))
    print(round(nMAE(pred['true'], pred['pred']),2))
    print(round(RMSE(pred['true'], pred['pred']),2))
    print(round(MAE(pred['true'], pred['pred']),2))

print(anser)


# In[ ]:


# print(round(nRMSE(pred['true'], pred['pred']),2))
# print(round(nMAE(pred['true'], pred['pred']),2))
# print(round(RMSE(pred['true'], pred['pred']),2))
# print(round(MAE(pred['true'], pred['pred']),2))

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


xtick = int(len(test_data['TIME_TO_INTERVAL'])/96)

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
    title='預測結果',
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




