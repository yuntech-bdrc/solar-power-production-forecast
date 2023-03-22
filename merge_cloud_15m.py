#!/usr/bin/env python
# coding: utf-8

# # <font color=#0000FF>將整數轉換為時間類型</font>

# In[1]:


# 定義一個函數將整數轉換為時間類型並返回格式化後的字符串(ex:0->00:00:00)
def format_time(i):
    # 將整數轉時間格式
    time = datetime.timedelta(hours=i)
    # 將時間類型轉成指定字串格式回傳
    #zfill(8)确保字串長度為8，（例如，'01:23:45'）
    return str(time).zfill(8)


# # <font color=#0000FF>將雲資料擴增4倍儲存到空的Dataframe</font>

# In[2]:


def cold_data_15m(df_cloud,cloudlen):
   #創建空的dataframe
    cold_new_data = [{'TIME_TO_INTERVAL'},{'lon'},{'lat'},{'cloud'},{'low'},{'mid'},{'hig'}]
    # 將 cold_new_data 中的元素轉換為列表
    cols = [list(x)[0] for x in cold_new_data]
    # 轉dataframe
    cold_new_data = pd.DataFrame(columns=cols)

    #整個TIME_TO_INTERVAL時間欄(col)
    time_all = df_cloud.iloc[1:,0]  
    for j in range(0,cloudlen-1):
        #cloudlen-1等於最後一筆的位索引,當讀到最後一筆的索引跳出
        if j == cloudlen-1:
            break
        else:
            df_datatime = pd.date_range(start=time_all[j+1], periods=4, freq='15min')
            #第一筆位置為0~3,第一筆後要從位置3之後開始存資料
            if j == 0:
                x = j   
            else:
                x = j+(j*3)
            # 添加第一個row數據(00:00:00)
            cold_new_data.loc[x, 'TIME_TO_INTERVAL'] = df_datatime[0]
            cold_new_data.loc[x, 'lon'] = df_cloud.iloc[j+1,1]
            cold_new_data.loc[x, 'lat'] = df_cloud.iloc[j+1,2]
            cold_new_data.loc[x, 'cloud'] =  df_cloud.iloc[j+1,3]
            cold_new_data.loc[x, 'low'] =  df_cloud.iloc[j+1,4]
            cold_new_data.loc[x, 'mid'] =  df_cloud.iloc[j+1,5]
            cold_new_data.loc[x, 'hig'] =  df_cloud.iloc[j+1,6]

            # 添加第2到4個row數據(00:15:00~00:45:00)
            for i in range(1,4):
                cold_new_data.loc[x+i, 'TIME_TO_INTERVAL'] = df_datatime[i]
    return cold_new_data


# # <font color=#0000FF>處理雲資料的時間欄位</font>

# In[3]:


import pandas as pd
import numpy as np
import time
import datetime

#history_15的csv檔
df_X = pd.read_csv(f'./dataset/solar_plant_newbig_sort(history_15m).csv')
dfXlen = len(df_X)

#篩選時間大於等於2022/2/1小於等於2022/10/31(因為雲資料集只有該區間時間資料)
dfXlen_result = df_X[(df_X['TIME_TO_INTERVAL'] >= '2022-02-01 00:00:00') 
                & (df_X['TIME_TO_INTERVAL'] < '2022-11-01 00:00:00')]


#雲資料
df_cloud = pd.read_csv(f'./cloud_data/cloud_datas_hour.csv', header=None)
#1.處理雲資料的時間('time')col
#1-1.將列轉換為整數類型,同時忽略任何轉換錯誤（使用 errors='coerce' 參數）
df_cloud[0][1:] = pd.to_numeric(df_cloud[0][1:], errors='coerce').astype('Int64')
#1-2.將指定列的所有整數值轉換為時間類型並覆蓋原始值(ex:0->00:00:00),呼叫format_time函數
df_cloud[0][1:] = df_cloud[0][1:].apply(lambda x: format_time(x))

#2.將('time'欄和'TIME_TO_INTERVAL'合併)
df_cloud[0] = df_cloud[7].astype(str) + ' ' + df_cloud[0].astype(str)
#2-1.刪除原版的'TIME_TO_INTERVAL'的col，並將結果存回 DataFrame
df_cloud.drop(7, axis=1, inplace=True)
#2-2.取代第一個col的row值
df_cloud.iat[0, 0] = 'TIME_TO_INTERVAL'

#雲資料長度
cloudlen = len(df_cloud)

#呼叫cold_data_15m函數進行資料擴增
cold_new_data = cold_data_15m(df_cloud,cloudlen)

#轉DataFrame做線性補值並合併history_15的csv檔
df_cold = pd. DataFrame({
  'lon': cold_new_data['lon'].astype(float),
  'lat': cold_new_data['lat'].astype(float),
  'cloud': cold_new_data['cloud'].astype(float),
  'low': cold_new_data['low'].astype(float),
  'mid': cold_new_data['mid'].astype(float),
  'hig': cold_new_data['hig'].astype(float)
})


df_cold = df_cold.interpolate(method ='linear',limit_area='inside')#limit_area='inside'僅填充被有效值包圍的 NaN（插值）

#將原有索引替換為從0開始的新索引
dfXlen_result = dfXlen_result.reset_index(drop=True).reset_index(drop=True)
df_cold = df_cold.reset_index(drop=True)

#合併history_15的csv檔
df_all = pd.concat([dfXlen_result,df_cold],axis=1)

#存成csv
df_all.to_csv(f'./dataset/solar_plant(history_cloud_15m).csv', index=None)


# In[4]:


df_all

