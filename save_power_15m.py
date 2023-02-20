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


# # 抓取資料庫全部資料

# In[4]:


# 
# def get_power():
#     tablename = 'MG1_PV'
#     # result = client.query(f'SELECT * FROM {tablename} where Time >= 2022-05-20') 
#     result = client.query(f'SELECT * FROM {tablename}') 
#     data = list(result.get_points())
#     data = pd.DataFrame(data)
#     merge_data=pd.DataFrame()
#     num = 0
#     fre = 0
#     for i in range(len(data)):
#         num += 1
#         if(num==50000):
#             number = num*fre
#             merge_data.to_csv(f"power_data/original/{tablename}_{number}.csv", index=False) 
#             num = 0
#             fre += 1
#             merge_data=pd.DataFrame()
#         else:
#             merge_data = pd.concat([merge_data,data.loc[i:i]],axis=0,ignore_index=True)
#     number = number+50000
#     merge_data.to_csv(f"power_data/original/{tablename}_{number}.csv", index=False) 
# get_power()


# # 抓取最新資料存入資料夾檔案中

# In[5]:


#抓取資料夾內原始資料
def get_file_number():
    path=".\\power_data\\original"
    filenames = os.listdir(path)
    number = []
    for i in range(len(filenames)):
        if(filenames[i] == 'save'):
            continue
        filenames_row = filenames[i].split('_')
        filenames_row = filenames_row[-1].split('.')
        number.append(filenames_row[0])
    number = sorted(number, key=int)
    return number


# In[6]:


# last = pd.read_csv(f'./power_data/original/MG1_PV_4750000.csv', low_memory=False) 
# last_time = pd.to_datetime(last['Time'][-1:].values[0])
# client = InfluxDBClient('120.107.146.56', 8086, 'ncue01', 'Q!A@Z#WSX', 'MG1') 
# tablename = 'MG1_PV'
# result = client.query(f"SELECT * FROM {tablename} where Time >= '{last_time}'-8h") 
# data =  list(result.get_points())
# data = pd.DataFrame(data)
# data


# In[7]:


#根據資料夾內最後一筆時間抓取最新資料
def get_power_2(number,date):
    client = InfluxDBClient('120.107.146.56', 8086, 'ncue01', 'Q!A@Z#WSX', 'MG1') 
    tablename = 'MG1_PV'
    result = client.query(f"SELECT * FROM {tablename} where time >= '{date}' - 8h") 
    data =  list(result.get_points())
    data = pd.DataFrame(data)
    last = pd.read_csv(f'./power_data/original/MG1_PV_{number[-1]}.csv', low_memory=False) 
    new = pd.DataFrame()
    new_bool = False
    for i in tqdm(range(len(data))):#當資料筆數到達50000筆後存下一個csv檔
        if(len(last) < 50000):
            last = pd.concat([last,data.loc[i:i]],axis=0)
        else:
            new_bool = True
            new = pd.concat([new,data.loc[i:i]],axis=0)
            new_number = int(number[-1])+50000
    last.to_csv(f"./power_data/original/MG1_PV_{number[-1]}.csv", index=False)
    if(new_bool):
        new.to_csv(f"./power_data/original/MG1_PV_{new_number}.csv", index=False)


# In[8]:


# number = get_file_number()
# last = pd.read_csv(f'./power_data/original/MG1_PV_{number[-1]}.csv', low_memory=False) 
# last_time = pd.to_datetime(last['Time'][-1:].values[0])
# #獲得最新資料
# get_power_2(number,last_time)


# In[9]:


#將資料轉成每15分鐘1筆
def bulid_15minute_data(data_raw):
    data_raw['TIME_TO_INTERVAL'] = pd.to_datetime(data_raw['TIME_TO_INTERVAL'])
    data_raw_2 = data_raw.groupby(pd.Grouper(key="TIME_TO_INTERVAL",freq='15min', origin='start')).mean().reset_index()
    return data_raw_2
# new_number = get_file_number()

def merge_old_new_data15(new_number):
    merge_data=pd.DataFrame()
    for i in new_number[-2:]:#-2為讀MG1_PV_{i}.csv倒數兩筆csv，如許久未跑這支程式，須調整數值，讀取倒數數筆csv檔
        last = pd.read_csv(f'./power_data/original/MG1_PV_{i}.csv', low_memory=False) 
        merge_data = pd.concat([merge_data,last],axis=0,ignore_index=True)
    merge_data = merge_data.rename(columns={'Time':'TIME_TO_INTERVAL'})
    merge_data = merge_data.drop_duplicates(['TIME_TO_INTERVAL'], keep="last").reset_index(drop=True)
    merge_data = merge_data.sort_values(by=['TIME_TO_INTERVAL']).reset_index(drop=True)
    for i in range(len(merge_data)):
        row = merge_data.loc[i:i].reset_index(drop=True)
        result = time.strptime(row['TIME_TO_INTERVAL'][0], "%Y-%m-%d %H:%M:%S") 
        result = time.strftime("%M:00",result)
#         print(result)
        if((result=='00:00')|(result=='15:00')|(result=='30:00')|(result=='45:00')):
            break      
    row['TIME_TO_INTERVAL'] = pd.to_datetime(row['TIME_TO_INTERVAL'])
    merge_data['TIME_TO_INTERVAL'] = pd.to_datetime(merge_data['TIME_TO_INTERVAL'])
    mask = (merge_data['TIME_TO_INTERVAL'] >= row['TIME_TO_INTERVAL'][0])
    merge_data = merge_data[mask] 
    merge_data = bulid_15minute_data(merge_data)  
    old_15_power = pd.read_csv(f'./power_data/merge_alldata_15.csv', low_memory=False) 
    old_15_power['TIME_TO_INTERVAL'] = pd.to_datetime(old_15_power['TIME_TO_INTERVAL'])
    merge_data = pd.concat([merge_data,old_15_power],axis=0,ignore_index=True)
    merge_data = merge_data.drop_duplicates(['TIME_TO_INTERVAL'], keep="last").reset_index(drop=True)
    merge_data = merge_data.sort_values(by=['TIME_TO_INTERVAL'])
    merge_data.to_csv(f"./power_data/merge_alldata_15.csv", index=False)


# # 讀取資料和模型，做績效

# In[10]:


minute = pd.to_datetime(datetime.datetime.today())-datetime.timedelta(minutes=15)
minute


# In[11]:


def split_data(data,target_day):
    power_list=['pre_Power_15','pre_Power_30','pre_Power_45']
    Radiation_list=['pre_Radiation_15','Radiation_0','next_Radiation_15']
    data_merge = data.copy()
    row = target_day.copy()
    #建立三個表
    data_power = pd.DataFrame()
    data_Radiation = pd.DataFrame()
    data_2 = pd.DataFrame()  
    data_power = data_merge[data_merge['date'].isin(row['date'])]
    
    row_time = pd.to_datetime(row['TIME_TO_INTERVAL'].values[0])
    pre_time = [row_time-datetime.timedelta(minutes=15),
                row_time-datetime.timedelta(minutes=30),
                row_time-datetime.timedelta(minutes=45)]
    for i in range(len(pre_time)):
        pre_time[i] = pre_time[i].strftime("%Y-%m-%d %H:%M:%S")
       
    
    row_date = pd.to_datetime(row['date'].values[0])
    next_date = [row_date,
                 row_date+datetime.timedelta(days=1)]   
#     print(next_date)
    for i in range(len(next_date)):
        next_date[i] = next_date[i].strftime("%Y-%m-%d")
    #print(next_date)       
    pre_Radiation = [row_time-datetime.timedelta(minutes=15),
                    row_time,
                    row_time+datetime.timedelta(minutes=15)]
    for i in range(len(pre_Radiation)):
        pre_Radiation[i] = pre_Radiation[i].strftime("%Y-%m-%d %H:%M:%S")
        
    data_merge['date'] = data_merge['date'].apply(lambda x: x.strftime('%Y-%m-%d'))     
    data_Radiation = data_merge[data_merge['date'].isin(next_date)]
#     print(data_Radiation)
    for h in range(0,3): 

        data_power_2 = data_power[data_power['TIME_TO_INTERVAL'].isin([pre_time[h]])].reset_index(drop=True)  
        data_Radiation_2 = data_Radiation[data_Radiation['TIME_TO_INTERVAL'].isin([pre_Radiation[h]])].reset_index(drop=True)

#         print(data_power)
#         print(data_Radiation_2)
    
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


# In[12]:


def merge_power_weather():
    power_data = pd.read_csv(f"./power_data/merge_alldata_15.csv", low_memory=False) 
    power_data['TIME_TO_INTERVAL'] = pd.to_datetime(power_data['TIME_TO_INTERVAL'])
    mask = (power_data['TIME_TO_INTERVAL'] >= (pd.to_datetime(datetime.datetime.today())-datetime.timedelta(hours=1)))
    power_data = power_data[mask]
#     print(power_data)
    power_data = power_data.rename(columns={'kP':'Power'})
    power_data['hour'] = pd.to_datetime(power_data['TIME_TO_INTERVAL']).dt.hour
    power_data['date'] = pd.to_datetime(power_data['TIME_TO_INTERVAL']).dt.date
    power_data = power_data[['TIME_TO_INTERVAL','date','hour','Power']]
    power_data = power_data.dropna(subset=['Power']).reset_index(drop=True)
    power_data = power_data.drop_duplicates(['TIME_TO_INTERVAL'], keep="last").reset_index(drop=True)
    weatherdata = pd.read_csv('dataset/solar_汙水廠(history).csv')
    weatherdata = weatherdata.rename(columns={'Radiation(SDv3)(IBM)':'Radiation(SDv3)(TWC)',
                            'WeatherType(IBM)':'WeatherType(TWC)',
                            'WeatherType(pred)(IBM)':'WeatherType(pred)(TWC)',
                            'Radiation(MSM)':'Radiation(SDv3)(MSM)'})
    weatherdata['hour'] = pd.to_datetime(weatherdata['TIME_TO_INTERVAL']).dt.hour
    weatherdata['date'] = pd.to_datetime(weatherdata['TIME_TO_INTERVAL']).dt.date
    weatherdata = weatherdata[['hour','date','Radiation','ClearSkyRadiation','Radiation(SDv3)(CWB)',
         'Radiation(SDv3)(TWC)','Radiation(SDv3)(OWM)','Radiation(SDv3)(MSM)','Radiation(today)(CWB)',
         'Radiation(today)(IBM)','Radiation(today)(OWM)']]
    merge_data = pd.merge(power_data,weatherdata,on=['date','hour'],how='inner')
    merge_data['minute'] = pd.to_datetime(merge_data['TIME_TO_INTERVAL']).dt.minute
    merge_data = merge_data.drop_duplicates(['TIME_TO_INTERVAL'], keep="last").reset_index(drop=True)
    
    pre_datas = pd.DataFrame()
    for i in range(len(merge_data)):
        target_day = merge_data.loc[i:i].reset_index(drop=True)
        pre_data = split_data(merge_data,target_day)
        pre_datas = pd.concat([pre_datas,pre_data],axis=0)
    pre_datas = pre_datas.fillna(0)    
    pre_datas.reset_index(drop=True,inplace=True)
    merge_data = merge_data.merge(pre_datas, how='left', left_index=True, right_index=True)
#     print(merge_data)
    return merge_data


# In[13]:


def pred_power_15(merge_data):
    train_15 = pd.read_csv(f'power_data/merge_weather_power_for_train15(cwb).csv')
    train_data = train_15.copy()
    feature_data = ['pre_Power_15','next_Radiation_15','pre_Power_30']
    train_x = train_data[feature_data]
    train_y = train_data[['Power']]
    
    test_data = merge_data
    #獲得訓練集X的最大最小值，並正規劃測試資料
    scaler_x = MinMaxScaler()
    scaler_x.fit(train_x[feature_data])
    test_data = scaler_x.transform(test_data[feature_data])
#     print(test_data)
    #獲得訓練集y的最大最小值
    scaler_y = MinMaxScaler()
    scaler_y.fit(train_y[['Power']])
    #載入模型並預測+反正規劃
    model = joblib.load(f'model/15_minute/rvm_pred(cwb)(old_data).pkl')
    pred_y = model.predict(test_data)
    pred_y = pred_y.reshape(-1,1)
    pred_y = scaler_y.inverse_transform(pred_y)
    pred_y = pred_y.reshape(-1)
#     print(pred_y)
    #將預測資料和預測時間組成表格
    pred = pd.DataFrame()
    pred['TIME_TO_INTERVAL'] = pd.to_datetime(merge_data['TIME_TO_INTERVAL'])+datetime.timedelta(minutes=30)
    pred['pred'] = pred_y
    pred = pred.reset_index(drop=True)
    return pred


# In[14]:


def save_to_database(pred):
    client = InfluxDBClient('120.107.146.56', 8086, 'ncue01', 'Q!A@Z#WSX')
    # 目前有哪些資料庫名稱
    exist = client.get_list_database()
    number=0
    for i in range(len(exist)):
        if(exist[i] =={'name': 'Minute_Ahead_Pred'}):        
            number+=1
    if(number==1):
        client = InfluxDBClient('120.107.146.56', 8086, 'ncue01', 'Q!A@Z#WSX','Minute_Ahead_Pred')
    else:
        # 創建資料庫
        client.create_database('Minute_Ahead_Pred') 


    # 資料 (不用寫時間，InfluxDB會自動生成時間戳記)
    for i in range(len(pred)):
        target_day = pred.loc[i:i].reset_index(drop=True)
        data = [
            {
                "measurement": "汙水場",
                "tags": {
                    "UpdateTime": datetime.datetime.now(),
                },
                "time": pd.to_datetime(target_day['TIME_TO_INTERVAL'].values[0], format='%Y%m%dT%H:%M:%SZ'),
                
                "fields": {
                    "D_power":target_day['pred'].values[0],
                }
            }
        ]

        # 寫入數據，同時創建表
        client.write_points(data) 
    return "ok"


# In[20]:


localtime = time.localtime()
result = time.strftime("%M:%S", localtime)
result


# # 從每小時的05分開始執行，並且每15分鐘執行一次

# In[21]:


while(True):
    start_time = time.time()
    localtime = time.localtime()
    result = time.strftime("%M:%S", localtime)
    #0~20分執行的話會報錯
#     if((result=='00:00')|(result=='15:00')|(result=='30:00')|(result=='45:00')):
    #獲得新資料並整合至原始資料中
    number = get_file_number()
    last = pd.read_csv(f'./power_data/original/MG1_PV_{number[-1]}.csv', low_memory=False) 
    last_time = pd.to_datetime(last['Time'][-2:-1].values[0])
    print(last_time)
    get_power_2(number,last_time)
    #將新資料整合成15分鐘，並和舊資料合併
    new_number = get_file_number()
    merge_old_new_data15(new_number)
    #將15分鐘資料和天氣合併，並切割好欄位資料
    merge_data = merge_power_weather()
    merge_data = merge_data[-2:-1]
#     print(merge_data)
    pred = pred_power_15(merge_data)
    pred['pred'] = pred['pred'].where(pred['pred'] >= 0, 0)
    save_to_database(pred)
    print('OkOk')
    end_time = time.time()
    finish = end_time - start_time
    print(finish)
    time.sleep(900-finish)
#     else:
#         m,s = result.strip().split(":")
#         start_time = int(m)*60+int(s)
#         time.sleep(4800-start_time)


# In[ ]:


localtime = time.localtime()
localtime


# In[ ]:


row['TIME_TO_INTERVAL'][0]


# In[ ]:


merge_data = pd.read_csv(f'./power_data/original/MG1_PV_4150000.csv', low_memory=False) 
merge_data = merge_data.rename(columns={'Time':'TIME_TO_INTERVAL'})
merge_data = merge_data.sort_values(by=['TIME_TO_INTERVAL'])
for i in range(len(merge_data)):
    row = merge_data.loc[i:i].reset_index(drop=True)
#     row['TIME_TO_INTERVAL'] = pd.to_datetime(row['TIME_TO_INTERVAL']
    result = time.strptime(row['TIME_TO_INTERVAL'][0], "%Y-%m-%d %H:%M:%S") 
    result = time.strftime("%M:%S",result)
result    


# # 舊資料和新資料合併，並轉成15分鐘

# In[ ]:


# def bulid_15minute_data(data_raw):
#     data_raw['TIME_TO_INTERVAL'] = pd.to_datetime(data_raw['TIME_TO_INTERVAL'])
#     data_raw_2 = data_raw.groupby(pd.Grouper(key="TIME_TO_INTERVAL",freq='15min', origin='start')).mean().reset_index()
#     return data_raw_2
# def merge_file():
#     #抓取舊的POWER資料
#     path_1="C:\\Users\\IDSL\\Desktop\\G.Z\\太陽能\\太陽能發電\\天氣資料爬蟲與合併\\power_data\\MG1_PV"
#     filenames = os.listdir(path_1)
#     merge_1 = pd.DataFrame()
#     for i in tqdm(range(len(filenames))):
#         if(filenames[i] == 'save'):
#             continue
#         file_data = pd.read_csv(f'./power_data/MG1_PV/{filenames[i]}', low_memory=False) 
#         merge_1 = pd.concat([merge_1,file_data],axis=0,ignore_index=True)
#     merge_1 = merge_1.rename(columns={'Time':'TIME_TO_INTERVAL'})
#     merge_1 = merge_1.sort_values(by=['TIME_TO_INTERVAL'])

#     #抓取新的POWER資料
#     path_2="C:\\Users\\IDSL\\Desktop\\G.Z\\太陽能\\太陽能發電\\天氣資料爬蟲與合併\\power_data\\original"
#     filenames = os.listdir(path_2)
#     merge_2 = pd.DataFrame()
#     for i in tqdm(range(len(filenames))):
#         if(filenames[i] == 'save'):
#             continue
#         file_data = pd.read_csv(f'./power_data/original/{filenames[i]}', low_memory=False) 
#         merge_2 = pd.concat([merge_2,file_data],axis=0,ignore_index=True)
#     merge_2 = merge_2.rename(columns={'Time':'TIME_TO_INTERVAL'})
#     merge_2 = merge_2.sort_values(by=['TIME_TO_INTERVAL'])
  
#     #合併新舊資料
#     merge = pd.concat([merge_1,merge_2],axis=0,ignore_index=True)
#     merge = merge.sort_values(by=['TIME_TO_INTERVAL'])
#     merge = bulid_15minute_data(merge)  
#     return merge
# merge = merge_file()


# In[ ]:


merge
merge.to_csv(f"./power_data/merge_alldata_15.csv", index=False)


# In[ ]:


merge = merge.rename(columns={'kP':'Power'})
merge = merge.dropna(subset=['Power'])
merge.to_csv(f"./power_data/merge_TEST.csv", index=False)


# In[ ]:


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


xtick = int(len(merge['TIME_TO_INTERVAL'])/96)

fig_line = go.Figure()
fig_line.add_trace(go.Scatter(y = merge['Power'], x=merge['TIME_TO_INTERVAL'],
                    mode='lines',
                    name='真實值',
                    line={'dash': 'dash'},
                    line_color= '#1f77b4'))

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


data = pd.DataFrame(data)
data = data.rename(columns={'Time':'TIME_TO_INTERVAL'})
#data = bulid_hour_data(data)
data.to_csv('power_data/original/original_data.csv')


# In[ ]:


aaa = pd.read_csv('power_data/original/original_data.csv')


# In[ ]:


aaa


# In[ ]:




