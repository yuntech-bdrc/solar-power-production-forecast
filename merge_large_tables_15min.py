#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install schedule


# In[2]:


import pandas as pd
import numpy as np
#排程
import schedule
import time
from pandas import Timestamp
import datetime


# In[3]:


df_X = pd.read_csv(f'dataset/solar_汙水廠(history).csv', header=None)
dfXlen = len(df_X)
#從df_X中選取所有列的資料[從第一列開始,篩選到DataFrame物件的最後一列,保留DataFrame物件中所有的列]
df_X = df_X.iloc[0:dfXlen,:]
df_X = df_X.drop(columns=[5, 9, 13, 14, 20, 21, 22])#刪除IBM欄位
df_X


# # <font color=#0000FF>取大表(solar_汙水廠(history))進行4倍擴增</font>

# In[4]:


#整個時間欄(col)
time_all = df_X[0].values

df_X1 = df_X.iloc[0,:].to_frame().T

for j in range(1,dfXlen):
    #每隔15分鐘產生一個時間，共產生4個
    df_datatime = pd.date_range(start=time_all[j], periods=4, freq='15min')
    #第j行(讀取每一個row)
    df_X_original =df_X.loc[j,:]

    for i in range(1,4):
        #將就的替換成新的 ex:2020-06-01 20:00:00 -> 2020-06-01 20:15:00
        df_X_new = df_X_original.replace(str(df_datatime[0]), str(df_datatime[i]))
        #將需要做線性的欄位值都設為空,以利後續做concat
        df_X_new[[1,2,3,4,6,7,8,10,17,18,19,23,24,25]] = np.nan
    
        if i==1 and j ==1:
            #to_frame().T為轉置
            df_X1 = pd.concat([df_X,df_X_new.to_frame().T],axis=0, ignore_index=True)
        else:
            df_X1 = pd.concat([df_X1,df_X_new.to_frame().T],axis=0, ignore_index=True)
        
# df_X1 = df_X1.sort_values(by=0)
df_X1.to_csv(f'dataset/solar_汙水廠_newbig(history).csv', index=None,header=False)


# # <font color=#0000FF>將solar_汙水廠_newbig(history)檔依時間排序</font>

# In[5]:


df_X = pd.read_csv(f'dataset/solar_汙水廠_newbig(history).csv') 
df_X = df_X.sort_values(by='TIME_TO_INTERVAL')
df_X.to_csv(f'dataset/solar_汙水廠_newbig_sort(history).csv', index=None)


# # <font color=#0000FF>讀取15分鐘csv檔(該檔尚未做線性)</font>

# In[6]:


df_X = pd.read_csv(f'./dataset/solar_汙水廠_newbig_sort(history).csv')


# # <font color=#0000FF>轉DataFrame做線性補值</font>

# In[7]:


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

# 線性進行插值(備注:df_X2為名目資料,不需要做線性補值)
df_X1 = df_X1.interpolate(method ='linear',limit_area='inside')#limit_area='inside'僅填充被有效值包圍的 NaN（插值）
df_X3 = df_X3.interpolate(method ='linear',limit_area='inside')

#合併資料欄
df_all = pd.concat([df_TIME_TO_INTERVAL,df_X1,df_X2,df_X3],axis=1)
df_all
#存成csv
df_all.to_csv(f'./dataset/solar_汙水廠_newbig_sort(history_15m).csv', index=None)


# # <font color=#0000FF>每小時又三分自動讀取大表擴增</font>

# In[8]:


def Auto_merge_15m():
    #取現在時間
    now = datetime.datetime.now() 
    #讀檔
    solar_new = pd.read_csv(f'dataset/solar_汙水廠(new).csv', header=None)
    history_15m = pd.read_csv(f'./dataset/solar_汙水廠_newbig_sort(history_15m).csv', header=None)
    dfXlen = len(solar_new)
    solar_new = solar_new.iloc[0:dfXlen,:]
    solar_new = solar_new.drop(columns=[5, 9, 13, 14, 20, 21, 22])#刪除IBM欄位
    
    #取solar_汙水廠(new)的TIME_TO_INTERVAL欄值(col)
    # solar_new_noheader格式=2023/1/26 00:00
    solar_new_noheader = solar_new[0]
    #將 solar_new_noheader 中的TIME_TO_INTERVAL資料解析為 datetime 物件且從第一筆開始取資料(不含TIME_TO_INTERVAL標題欄)
    solar_new_noheader[1:] = pd.to_datetime(solar_new_noheader[1:], format='%Y/%m/%d %H:%M')
    # solar_new_noheader格式= 2023-01-26 00:00:00
    solar_new_noheader = solar_new_noheader[1:]

    # 取代原本的TIME_TO_INTERVAL欄的值(因solar_new_noheader沒有TIME_TO_INTERVAL標題欄)
    solar_new_noheader.loc[0] ='TIME_TO_INTERVAL'
    solar_new[0] = solar_new_noheader

    #取solar_汙水廠(new)的第一筆TIME_TO_INTERVAL
    new_first_col = solar_new[0]
    new_first_row = new_first_col[1]
    #因new_first_row是Timestamp,需轉時間字串
    new_first_row = new_first_row.strftime('%Y-%m-%d %H:%M:%S')

    #取solar_汙水廠_newbig_sort(history_15m)的最後一筆TIME_TO_INTERVAL
    history_row = history_15m.iloc[-1]
    history_col = history_row[0]

    #創建空的dataframe
    solar_new_dataframe = [
        {'TIME_TO_INTERVAL'},
        {'Power'},
        {'Radiation'},
        {'ClearSkyRadiation'},
        {'Radiation(SDv3)(CWB)'},
        {'Radiation(SDv3)(OWM)'},
        {'Radiation(MSM)'},
        {'Radiation(today)(CWB)'},
        {'Radiation(today)(OWM)'},
        {'WeatherType(CWB)'},
        {'WeatherType(pred)(CWB)'},
        {'WeatherType(OWM)'},
        {'WeatherType(pred)(OWM)'},
        {'ApparentTemperature(pred)[CWB]'},
        {'Temperature(pred)[CWB]'},
        {'RelativeHumidity(pred)[CWB]'},
        {'FeelsLikeTemperature(pred)[OWM]'},
        {'Temperature(pred)[OWM]'},
        {'RelativeHumidity(pred)[OWM]'}
    ]

    # 將 solar_new_dataframe 中的元素轉換為列表
    cols = [list(x)[0] for x in solar_new_dataframe]

    # 創建空的 dataframe
    solar_new_dataframe = pd.DataFrame(columns=cols)

    #假設新的一小時大表第一筆資料時間大於15分鐘表的最後一筆資料
    if new_first_row > history_col:
        #整個TIME_TO_INTERVAL時間欄(col)
        time_all = solar_new.iloc[1:,0]  
        
        for j in range(0,dfXlen-1):
            #每隔15分鐘產生一個時間，共產生4個
            time_all_new = time_all.iloc[j]
            time_str = time_all_new.strftime('%Y-%m-%d %H:%M:%S')
            #df_datatime[0]=00:00,df_datatime[1]=00:15,df_datatime[2]=00:30,df_datatime[3]=00:45
            df_datatime = pd.date_range(start=time_str, periods=4, freq='15min')
            #第j+1行(一小時的),j欄是標題欄
            solar_new_original =solar_new.loc[j+1,:]
            #第一筆位置為0~3,第一筆後要從位置3之後開始存資料
            if j == 0:
                j = j   
            else:
                j = j+(j*3)
    
            # 添加第一行數據(EX:第一筆)
            solar_new_dataframe.loc[j, 'TIME_TO_INTERVAL'] = df_datatime[0]
            solar_new_dataframe.loc[j, 'Power'] = solar_new_original[1]
            solar_new_dataframe.loc[j, 'Radiation'] = solar_new_original[2]
            solar_new_dataframe.loc[j, 'ClearSkyRadiation'] = solar_new_original[3]
            solar_new_dataframe.loc[j, 'Radiation(SDv3)(CWB)'] = solar_new_original[4]
            solar_new_dataframe.loc[j, 'Radiation(SDv3)(OWM)'] = solar_new_original[6]
            solar_new_dataframe.loc[j, 'Radiation(MSM)'] = solar_new_original[7]
            solar_new_dataframe.loc[j, 'Radiation(today)(CWB)'] = solar_new_original[8]
            solar_new_dataframe.loc[j, 'Radiation(today)(OWM)'] = solar_new_original[10]
            solar_new_dataframe.loc[j, 'WeatherType(CWB)'] = solar_new_original[11]
            solar_new_dataframe.loc[j, 'WeatherType(pred)(CWB)'] =solar_new_original[12]
            solar_new_dataframe.loc[j, 'WeatherType(OWM)'] = solar_new_original[15]
            solar_new_dataframe.loc[j, 'WeatherType(pred)(OWM)'] = solar_new_original[16]
            solar_new_dataframe.loc[j, 'ApparentTemperature(pred)[CWB]'] =solar_new_original[17]
            solar_new_dataframe.loc[j, 'Temperature(pred)[CWB]'] = solar_new_original[18]
            solar_new_dataframe.loc[j, 'RelativeHumidity(pred)[CWB]'] = solar_new_original[19]
            solar_new_dataframe.loc[j, 'FeelsLikeTemperature(pred)[OWM]'] =solar_new_original[23]
            solar_new_dataframe.loc[j, 'Temperature(pred)[OWM]'] = solar_new_original[24]
            solar_new_dataframe.loc[j, 'RelativeHumidity(pred)[OWM]'] =solar_new_original[25]
            # 添加第2到4行數據(00:15~00:45)(EX:第二到四筆)
            for s in range(1, 4):
                solar_new_dataframe.loc[j+s, 'TIME_TO_INTERVAL'] = df_datatime[s]
                #天氣類型為名目資料,因此填入與一小時資料相同類型
                solar_new_dataframe.loc[j+s, 'WeatherType(CWB)'] = solar_new_original[11]
                solar_new_dataframe.loc[j+s, 'WeatherType(pred)(CWB)'] = solar_new_original[12]
                solar_new_dataframe.loc[j+s, 'WeatherType(OWM)'] = solar_new_original[15]
                solar_new_dataframe.loc[j+s, 'WeatherType(pred)(OWM)'] = solar_new_original[16]
           
        solar_new_dataframe.to_csv(f'dataset/solar_汙水廠_newbig(solar(new)_history).csv', index=None)
        print(now,'Data added successfully.')
    else:
        print(now,'NO new data.')

    # 讀取15分鐘csv檔(該檔尚未做線性)
    solar_history_new15m = pd.read_csv(f'dataset/solar_汙水廠_newbig(solar(new)_history).csv')

    # 轉DataFrame做線性補值
    # 創建數據集
    df_TIME_TO_INTERVAL =pd.DataFrame(solar_history_new15m['TIME_TO_INTERVAL'])
    df_X1 = pd. DataFrame({
      'Power': solar_history_new15m['Power'].astype(float),
      'Radiation': solar_history_new15m['Radiation'].astype(float),
      'ClearSkyRadiation': solar_history_new15m['ClearSkyRadiation'].astype(float),
      'Radiation(SDv3)(CWB)': solar_history_new15m['Radiation(SDv3)(CWB)'].astype(float),
      'Radiation(SDv3)(OWM)': solar_history_new15m['Radiation(SDv3)(OWM)'].astype(float),
      'Radiation(MSM)': solar_history_new15m['Radiation(MSM)'].astype(float),
      'Radiation(today)(CWB)': solar_history_new15m['Radiation(today)(CWB)'].astype(float),
      'Radiation(today)(OWM)': solar_history_new15m['Radiation(today)(OWM)'].astype(float),
    })
    df_X2 = pd. DataFrame({
      'WeatherType(CWB)': solar_history_new15m['WeatherType(CWB)'],
      'WeatherType(pred)(CWB)': solar_history_new15m['WeatherType(pred)(CWB)'],
      'WeatherType(OWM)	': solar_history_new15m['WeatherType(OWM)'],
      'WeatherType(pred)(OWM)': solar_history_new15m['WeatherType(pred)(OWM)']  
    })
    df_X3 = pd. DataFrame({
      'ApparentTemperature(pred)[CWB]': solar_history_new15m['ApparentTemperature(pred)[CWB]'].astype(float),
      'Temperature(pred)[CWB]': solar_history_new15m['Temperature(pred)[CWB]'].astype(float),
      'RelativeHumidity(pred)[CWB]': solar_history_new15m['RelativeHumidity(pred)[CWB]'].astype(float),
      'FeelsLikeTemperature(pred)[OWM]': solar_history_new15m['FeelsLikeTemperature(pred)[OWM]'].astype(float),
      'Temperature(pred)[OWM]': solar_history_new15m['Temperature(pred)[OWM]'].astype(float),
      'RelativeHumidity(pred)[OWM]': solar_history_new15m['RelativeHumidity(pred)[OWM]'].astype(float)  
    })

    # 線性進行插值
    df_X1 = df_X1.interpolate(method ='linear',limit_area='inside')#limit_area='inside'僅填充被有效值包圍的 NaN（插值）
    df_X3 = df_X3.interpolate(method ='linear',limit_area='inside')

    #合併資料欄
    df_all = pd.concat([df_TIME_TO_INTERVAL,df_X1,df_X2,df_X3],axis=1)
    df_all

    # 將新的DataFrame資料追加到原始CSV檔案中
    df_all.to_csv(f'./dataset/solar_汙水廠_newbig_sort(history_15m).csv', mode='a', header=False, index=False)
    
#排程
# at:每小時的第(n)分時執行
schedule.every(1).hours.at(":03").do(Auto_merge_15m)
while True:
    schedule.run_pending()


# In[ ]:




