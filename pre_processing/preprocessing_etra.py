# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:19:17 2023

@author: marca
"""

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt   


def pre_processing():
    
    scr_dim = [1024,768]
    subject_dirs = [x[0] for x in os.walk('dataset/ETRA/data')]
    
    for subject_dir in subject_dirs:
        records = [f for f in os.listdir(subject_dir) if f[-4:] == '.csv']
    
        for record in records:
 
          
            
            subject, trial, task, condition, stimulus = record.split('.')[0].split('_')
            
            print('Processing data for subject {sub_}, trial {tri_}, task {tas_}, condition {con_} and stimulus {sti_}'.format(sub_=subject,
                                                                                                                               tri_=trial,
                                                                                                                               tas_=task,
                                                                                                                               con_=condition,
                                                                                                                               sti_=stimulus))
            
            file = '{dir_}/{rec_}'.format(dir_ = subject_dir,
                                          rec_ = record)
            df = pd.read_csv(file)
     
            # Left and right positions are averaged
            x_ = np.mean(np.array([df['LXpix'].values, 
                                   df['RXpix'].values]), 
                         axis=0)
            y_ = np.mean(np.array([df['LYpix'].values, 
                                   df['RYpix'].values]), 
                         axis=0)
         
            # Create status array
            status = np.ones(len(x_), dtype=int)
 
            # Nan values
            nan_idx_x = (np.argwhere(np.isnan(x_)).flatten())
            status[nan_idx_x] = 0
           
            nan_idx_y = (np.argwhere(np.isnan(y_)).flatten()) 
            status[nan_idx_y] = 0
             
            # Creating clean df
            data_l = dict({
                 'gaze_x': x_,
                 'gaze_y': y_,
                     })
             
            df_clean = pd.DataFrame(data=data_l) 

            # Interpolating missing values
            df_clean = df_clean.interpolate(method='polynomial', order=1)
            
            # Out of bounds gaze samples 
            x_ = df_clean['gaze_x'].values
            status[x_>scr_dim[0]] = 0
            status[x_<0] = 0
            y_ = df_clean['gaze_y'].values
            status[y_>scr_dim[1]] = 0
            status[y_<0] = 0
            
            df_clean['status'] = status
            
            # Create array for timestamps
            ts = df['Time'].values / 1000
            ts -= ts[0]
            df_clean['timestamp'] = ts
            
            print('...done') 
            print('Saving dataframe...')
            
            filename = 'dataset/ETRA/parsed_data/{sub_}_{tri_}_{tas_}_{con_}_{sti_}.csv'.format(sub_=subject,
                                                                                                tri_=trial,
                                                                                                tas_=task,
                                                                                                con_=condition,
                                                                                                sti_=stimulus)
            df_clean.to_csv(filename, index=False) 
            #print(df_clean.head())
            print('...done')
            df_clean = pd.read_csv(filename)   
            
             
         
            
pre_processing()









