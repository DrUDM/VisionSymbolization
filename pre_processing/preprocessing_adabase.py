 # -*- coding: utf-8 -*-

import pandas as pd
import numpy as np 
import math 

import itertools 
from itertools import groupby
from operator import itemgetter

import os 
import matplotlib.pyplot as plt   





def pre_processing():
   
    ## To ensure positive values
    scr_dim = [4.5, 3.5]
    horizontal_offset = 1.5
    vertical_offset = 0.5
    
    ## Use only data samples with both right and left signals
    partial_ = True
    
    SUBJECT_IDX = list(range(30)) 
    data_folder = "dataset/ADABase/data"
   
    file_names = os.listdir(data_folder)
    file_names = list(filter(lambda x: x.endswith(".h5py"), file_names))
   
    print(file_names)
    for subject in SUBJECT_IDX:   
        file_name = file_names[subject]
        file_path = os.path.join(data_folder , file_name) 
        #with pd.HDFStore(file_path) as hdf:
            ## This prints a list of all group names:
            #print(hdf.keys()) 
        df_signal = pd.read_hdf(file_path, "SIGNALS", mode="r")
      
        stu_ = df_signal['STUDY'].values
        pha_ = df_signal['PHASE'].values
        lev_ = df_signal['LEVEL'].values
        labels = []
        for i in range(len(stu_)):
            labels.append('{stu}_{pha}_{lev}'.format(stu = stu_[i], 
                                                     pha = pha_[i], 
                                                     lev = lev_[i])) 
        res_labels = get_label_intervals(labels)
     
        for res_label in res_labels: 
            spl = res_label[0].split('_') 
            if 'n/a' not in spl: 
                label = res_label[0].replace('-','')
                start, end = res_label[1], res_label[2] 
                 
                print('Processing data for subject {sub_}, for task {label}'.format(sub_=subject,
                                                                                    label = label))
                try: 
                    gaze_r_x = df_signal['RIGHT_GAZE_POINT_ON_DISPLAY_AREA_X'].values[start:end+1]
                    gaze_l_x = df_signal['LEFT_GAZE_POINT_ON_DISPLAY_AREA_X'].values[start:end+1]
                    gaze_r_y = df_signal['RIGHT_GAZE_POINT_ON_DISPLAY_AREA_Y'].values[start:end+1]
                    gaze_l_y = df_signal['LEFT_GAZE_POINT_ON_DISPLAY_AREA_Y'].values[start:end+1]
                   
                    gaze_signals = dict({'right_x': gaze_r_x, 
                                         'left_x': gaze_l_x, 
                                         'right_y': gaze_r_y, 
                                         'left_y': gaze_l_y})
                    ## Get first and last indexes to keep in order to reduce data size
                    first_indexes, last_indexes = [], []
                    for k in gaze_signals.keys():
                        gaze_signal = gaze_signals[k]
                        gaze_signal[gaze_signal == np.inf] = np.nan
                         
                        f_idx = next(i for i in range(len(gaze_signal)) 
                                     if not np.isnan(gaze_signal[i]))
                        first_indexes.append(f_idx)
                        
                        l_idx = next(i for i in range(1, len(gaze_signal)) 
                                     if not np.isnan(gaze_signal[-i]))
                        last_indexes.append(len(gaze_signal) - l_idx) 
                    f_idx = min(first_indexes)
                    l_idx = max(last_indexes)
                
                    down_signals = dict()
                    for k in gaze_signals.keys():
                        gaze_signal = gaze_signals[k]
                        d_sig = gaze_signal[f_idx:l_idx:4]
                        down_signals.update({k: d_sig}) 
      
                    gaze_x_b = np.vstack((down_signals['right_x'], down_signals['left_x']))
                    gaze_x = []
                   
                    gaze_y_b = np.vstack((down_signals['right_y'], down_signals['left_y']))
                    gaze_y = []
                    
                    ## Fusion of left and right gaze signals 
                    for i in range(gaze_x_b.shape[1]):
                        ## Allow using only left or right data
                        if partial_:
                            if np.isnan(gaze_x_b[0,i]) and np.isnan(gaze_x_b[1,i]):
                                gaze_x.append(np.nan)
                            else:
                                gaze_x.append(np.nanmean(gaze_x_b[:,i]))
                                
                            if np.isnan(gaze_y_b[0,i]) and np.isnan(gaze_y_b[1,i]):
                                gaze_y.append(np.nan)
                            else:
                                gaze_y.append(np.nanmean(gaze_y_b[:,i]))
                        ##Allow using both left and right when available 
                        else:
                            if not np.isnan(gaze_x_b[0,i]) and not np.isnan(gaze_x_b[1,i]):
                                gaze_x.append(np.mean(gaze_x_b[:,i]))
                            else:
                                gaze_x.append(np.nan)
                                
                            if not np.isnan(gaze_y_b[0,i]) and not np.isnan(gaze_y_b[1,i]):
                                gaze_y.append(np.mean(gaze_y_b[:,i]))
                            else:
                                gaze_y.append(np.nan) 
                                
                    ## Add offset to have only positive values 
                    gaze_x = np.array(gaze_x) + horizontal_offset
                    gaze_y = np.array(gaze_y) + vertical_offset
                    
                    plt.plot(gaze_x, gaze_y, 
                             linewidth=.3)
                    plt.axis('equal')
                    plt.xlim([0, scr_dim[0]]) 
                    plt.ylim([0, scr_dim[1]]) 
                    plt.title(label)
                    plt.show()
                    plt.clf()
                    
                    ## Create fake timestamps 
                    ts = np.ones((len(gaze_x))) * 0.004
                    ts[0]=0
                    ts = np.cumsum(ts)
             
                    ## Create status array
                    status = np.ones(len(gaze_x), dtype=int) 
                    ## Spot nan values
                    nan_idx_x = (np.argwhere(np.isnan(gaze_x)).flatten())
                    status[nan_idx_x] = 0 
                    nan_idx_y = (np.argwhere(np.isnan(gaze_y)).flatten()) 
                    status[nan_idx_y] = 0
                    
                    ## Creating clean df
                    data_l = dict({
                         'gazeX': gaze_x,
                         'gazeY': gaze_y,
                             })
                     
                    df_clean = pd.DataFrame(data=data_l) 
    
                    ## Interpolate missing values
                    df_clean = df_clean.interpolate(method='polynomial', order=1)
                    
                    # Out of bounds gaze samples 
                    x_ = df_clean['gazeX'].values
                    status[x_>scr_dim[0]] = 0
                    status[x_<0] = 0
                    y_ = df_clean['gazeY'].values
                    status[y_>scr_dim[1]] = 0
                    status[y_<0] = 0
                    
                    ## Add status and ts
                    df_clean['status'] = status 
                    df_clean['timestamp'] = ts
                    
                    print('...done') 
                    print('Saving dataframe...')
                    ## Save to csv
                    filename = 'dataset/ADABase/parsed_data/{sub_}_{label}.csv'.format(sub_=subject,
                                                                                       label=label)
                    df_clean.to_csv(filename, index=False)  
                    print('...done')
                    
                except:
                    print('No gaze data')
        

def get_label_intervals(input):
    
    res = []
    idx = 0
     
    while idx < (len(input)):
        strt_pos = idx
        val = input[idx]
     
        ## Getting last position
        while (idx < len(input) and input[idx] == val):
            idx += 1
        end_pos = idx - 1 
        ## Appending in format [element, start, end position]
        res.append((val, strt_pos, end_pos))
    
    return res


if __name__ == '__main__': 
    pre_processing()
 









