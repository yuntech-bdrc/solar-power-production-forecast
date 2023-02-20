#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df_X = pd.read_csv(f'dataset/solar_汙水廠(history).csv', header=None)
dfXlen = len(df_X)
df_X = df_X.iloc[0:dfXlen,:]
df_X = df_X.drop(columns=[5, 9, 13, 14, 20, 21, 22])#刪除IBM欄位
df_X


# In[2]:


import numpy as np

#整個時間欄
time_all = df_X[0].values

df_X1 = df_X.iloc[0,:].to_frame().T
#-------------------------------------
#測試變空值
# df_X_original =df_X.loc[15300,:]
# print(df_X_original)
# df_X_original[[1,2,3,4,6,7,8,10,17,18,19,23,24,25]] = np.nan
# print(df_X_original)
#-------------------------------------

for j in range(1,dfXlen):
    #每隔15分鐘產生一個時間，共產生4個
    df_datatime = pd.date_range(start=time_all[j], periods=4, freq='15min')
    #第j行
    df_X_original =df_X.loc[j,:]

    for i in range(1,4):
        #將就的替換成新的 ex:2020-06-01 20:00:00 -> 2020-06-01 20:15:00
        df_X_new = df_X_original.replace(str(df_datatime[0]), str(df_datatime[i]))
        df_X_new[[1,2,3,4,6,7,8,10,17,18,19,23,24,25]] = np.nan
    
        if i==1 and j ==1:
            #to_frame().T為轉置
            df_X1 = pd.concat([df_X,df_X_new.to_frame().T],axis=0, ignore_index=True)
        else:
            df_X1 = pd.concat([df_X1,df_X_new.to_frame().T],axis=0, ignore_index=True)
        
# df_X1 = df_X1.sort_values(by=0)
df_X1.to_csv(f'dataset/solar_汙水廠_newbig(history).csv', index=None,header=False)


# In[3]:


df_X = pd.read_csv(f'dataset/solar_汙水廠_newbig(history).csv') 
df_X = df_X.sort_values(by='TIME_TO_INTERVAL')
df_X.to_csv(f'dataset/solar_汙水廠_newbig_sort(history).csv', index=None)


# <font color=#0000FF># 讀取15分鐘csv檔(該檔尚未做線性)</font>

# In[12]:


df_X = pd.read_csv(f'./dataset/solar_汙水廠_newbig_sort(history).csv')


# <font color=#0000FF># 轉DataFrame做線性補差值</font>

# In[13]:


# 創建數據集
df_TIME_TO_INTERVAL =pd.DataFrame(df_X['TIME_TO_INTERVAL'])
df_X1 = pd. DataFrame({
  'Power': df_X['Power'].astype(float),
  'Radiation': df_X['Radiation'].astype(float),
  'ClearSkyRadiation': df_X['ClearSkyRadiation'].astype(float),
  'Radiation(SDv3)(CWB)': df_X['Radiation(SDv3)(CWB)'].astype(float),
  'Radiation(SDv3)(OWM)': df_X['Radiation(SDv3)(OWM)'].astype(float),
  'Radiation(MSM)': df_X['Radiation(MSM)'].astype(float),
  'Radiation(today)(CWB)': df_X['Radiation(today)(CWB)'].astype(float),
  'Radiation(today)(OWM)': df_X['Radiation(today)(OWM)'].astype(float),
})
df_X2 = pd. DataFrame({
  'WeatherType(CWB)': df_X['WeatherType(CWB)'],
  'WeatherType(pred)(CWB)': df_X['WeatherType(pred)(CWB)'],
  'WeatherType(OWM)	': df_X['WeatherType(OWM)'],
  'WeatherType(pred)(OWM)': df_X['WeatherType(pred)(OWM)']  
})
df_X3 = pd. DataFrame({
  'ApparentTemperature(pred)[CWB]': df_X['ApparentTemperature(pred)[CWB]'].astype(float),
  'Temperature(pred)[CWB]': df_X['Temperature(pred)[CWB]'].astype(float),
  'RelativeHumidity(pred)[CWB]': df_X['RelativeHumidity(pred)[CWB]'].astype(float),
  'FeelsLikeTemperature(pred)[OWM]': df_X['FeelsLikeTemperature(pred)[OWM]'].astype(float),
  'Temperature(pred)[OWM]': df_X['Temperature(pred)[OWM]'].astype(float),
  'RelativeHumidity(pred)[OWM]': df_X['RelativeHumidity(pred)[OWM]'].astype(float)  
})

# 線性進行插值
df_X1 = df_X1.interpolate(method ='linear',limit_area='inside')#limit_area='inside'僅填充被有效值包圍的 NaN（插值）
df_X3 = df_X3.interpolate(method ='linear',limit_area='inside')

#合併資料欄
df_all = pd.concat([df_TIME_TO_INTERVAL,df_X1,df_X2,df_X3],axis=1)
df_all


# In[14]:


df_all.to_csv(f'./dataset/solar_汙水廠_newbig_sort(history_15m).csv', index=None)


# In[ ]:




