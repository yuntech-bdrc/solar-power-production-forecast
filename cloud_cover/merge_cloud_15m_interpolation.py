#!/usr/bin/env python
# coding: utf-8

# # <font color=#0000FF>每10分鐘雲資料重取樣為每15分鐘一筆資料</font>

# In[216]:


#將十分鐘一筆資料重取樣為15分鐘一筆。ffill方法用來填補缺失值，確保資料能夠被正確重取樣。
#然後使用isin方法檢查df2(15m)中的日期是否在df(10m)中存在，如果存在則將資料複製到df2中。最後，使用interpolate方法對df2進行線性插值。
import pandas as pd
import numpy as np

# 讀取資料
df = pd.read_csv(f'./cloud_data/new_obs_cloud.csv')

# 將date_time欄位轉換成datetime格式
df['date_time'] = pd.to_datetime(df['date_time'])

# 將date_time欄位設定為索引
df.set_index('date_time', inplace=True)

# 進行重取樣，將十分鐘一筆資料重取樣為15分鐘一筆
df = df.resample('15min').ffill()

# 產生15分鐘一筆的日期時間欄位
dates = pd.date_range(start=df.index[0], end=df.index[-1], freq='15min')

# 產生空的df2 
df2 = pd.DataFrame({'date_time': dates, 'lon': 120.560722, 'lat': 24.066583, 'low': np.nan, 'mid': np.nan, 'hig': np.nan, 
                    'cloud': np.nan})

# 將df中的資料複製到df2中
df2.loc[df2['date_time'].isin(df.index), ['low', 'mid', 'hig' ,'cloud']] = df[['low', 'mid', 'hig','cloud']].values

# 對df2進行線性插值
df2[['low', 'mid', 'hig','cloud']] = df2[['low', 'mid', 'hig','cloud']].interpolate()

# 輸出df2
df2.to_csv(f'./cloud_data/10m_to_15m_colud.csv', index=None)


# In[217]:


#history_15的csv檔
df_X = pd.read_csv(f'./dataset/solar_汙水廠_newbig_sort(history_15m).csv')
dfXlen = len(df_X)

#篩選時間大於等於2022/2/1小於等於2022/10/31(因為雲資料集只有該區間時間資料)
dfXlen_result = df_X[(df_X['TIME_TO_INTERVAL'] >= '2022-02-01 00:00:00') 
                & (df_X['TIME_TO_INTERVAL'] < '2022-11-01 00:00:00')]

#將原有索引替換為從0開始的新索引
dfXlen_result = dfXlen_result.reset_index(drop=True).reset_index(drop=True)
df2 = df2.reset_index(drop=True)

#合併history_15的csv檔
df_all = pd.concat([dfXlen_result,df2],axis=1)

#存成csv
df_all.to_csv(f'./dataset/10m_to_15m_colud.csv', index=None)


# In[ ]:




