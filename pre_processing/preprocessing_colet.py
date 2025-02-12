# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np 
import math 

import itertools 
from itertools import groupby
from operator import itemgetter
import bisect 

import os 
import matplotlib.pyplot as plt   





def pre_processing():
    
    SUBJECT_IDX = list(range(1, 48)) 
    data_folder = "dataset/COLET/data/"
    
    file_names = os.listdir(data_folder)
 
    for idx in SUBJECT_IDX:  
      #if idx==1:
        local = [f for f in file_names if f.split('_')[1]==str(idx)]
        
        for task in range(1, 5): 
            local_local = [f for f in local if f.split('_')[3]==str(task)]
            gaze = list(filter(lambda x: x.endswith("gaze.csv"), local_local))[0]
            blinks = list(filter(lambda x: x.endswith("blinks.csv"), local_local))[0]
            
            
            
            df = pd.read_csv(data_folder+gaze)
            df_blinks = pd.read_csv(data_folder+blinks)
            
            blink_s = blink_status(df, df_blinks)
            
            
            #print(df.head())
            df_x = df['norm_pos_x'].to_numpy()
            df_y = df['norm_pos_y'].to_numpy()
            confidence = df['confidence'].to_numpy()
            
            df_x[df_x<0]=np.nan
            df_x[df_x>1]=np.nan
            
            df_y[df_y<0]=np.nan
            df_y[df_y>1]=np.nan
            
            df_x[blink_s==0] = np.nan
            df_y[blink_s==0] = np.nan
            
            df_x[confidence<0.4] = np.nan
            df_y[confidence<0.4] = np.nan
            
            plt.plot(df_x, df_y)
            plt.ylim(0,1)
            plt.xlim(0,1)

            plt.show()
            plt.clf()
            
            plt.plot(confidence)
            plt.show()
            plt.clf()
    
    
    
def blink_status(df, df_blinks):
    
    timestamps = df['gaze_timestamp'].to_numpy()
    print(timestamps)
    status = np.ones((len(timestamps)))
    
    try:
        for i in range(len(df_blinks)):
            strt_ts = df_blinks.iloc[i]['start_timestamp']
            end_ts = df_blinks.iloc[i]['end_timestamp']
            
            str_idx = bisect.bisect_left(timestamps, strt_ts)
            end_idx = bisect.bisect_right(timestamps, end_ts)
            
            status[str_idx:end_idx+1] = 0
    
    except:
        return status
    
    return status
    
    




if __name__ == '__main__': 
    pre_processing()
 