# -*- coding: utf-8 -*-

import numpy as np
import ruptures as rpt 
import pandas as pd
import matplotlib.pyplot as plt 



def process(config, path, feature_records):
    '''
    

    Parameters
    ----------
    config : TYPE
        DESCRIPTION.
    path : TYPE
        DESCRIPTION.
    feature_records : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    ## For oculomotor features only
    if True:
        print('Segmenting oculomotor features...')
        oculomotor_feature_records = [feature_record for feature_record in feature_records
                              if feature_record.split('.')[0].split('_')[-1] == 'oculomotor']
        process_oculomotor(config, path, oculomotor_feature_records)
        
    ## For scanpath features only
    if True:
        print('Segmenting scanpath features...')
        scanpath_feature_records = [feature_record for feature_record in feature_records
                              if feature_record.split('.')[0].split('_')[-1] == 'scanpath']
        process_scanpath(config, path, scanpath_feature_records)
            
    ## For aoi sequence features 
    if True: 
        print('Segmenting aoi features...')
        aoi_feature_records = [feature_record for feature_record in feature_records
                              if feature_record.split('.')[0].split('_')[-1] == 'AoI']
        process_aoi(config, path, aoi_feature_records)
   
        
   
def process_aoi(config, path, feature_records, 
                display=False):
    '''
    

    Parameters
    ----------
    config : TYPE
        DESCRIPTION.
    path : TYPE
        DESCRIPTION.
    feature_records : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    nb_bkps = config['symbolization']['segmentation']['aoi']['nb_breakpoints']
    outpath = 'output/results/ETRA/segmentation/'
    
    for record in feature_records: 
        subject, trial, task, condition, stimulus, _ = record.split('.')[0].split('_')
        
        if (subject in config['data']['subject_set']
            and task in config['data']['task_set']
            and condition in config['data']['condition_set']):
            
            df = pd.read_csv(path+record)  
            name = record.split('.')[0] 
            ## Full signal segmentation
            signal = df.to_numpy()[:,1:]
     
            bkps=signal_segmentation(signal,
                                     nb_bkps,
                                     )
            if display:
                display_segmentation(signal, 
                                     bkps, 
                                     name)
            bkps.insert(0, 0)
            filename = '{out_}{name_}.npy'.format(out_=outpath, 
                                                  name_=name)
            np.save(filename, np.array(bkps))
            
            
def process_scanpath(config, path, feature_records, 
                     display=False):
    '''
    

    Parameters
    ----------
    config : TYPE
        DESCRIPTION.
    path : TYPE
        DESCRIPTION.
    feature_records : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    nb_bkps = config['symbolization']['segmentation']['scanpath']['nb_breakpoints']
    outpath = 'output/results/ETRA/segmentation/'
    
    for record in feature_records: 
        subject, trial, task, condition, stimulus, _ = record.split('.')[0].split('_')
        
        if (subject in config['data']['subject_set']
            and task in config['data']['task_set']
            and condition in config['data']['condition_set']):
            
            df = pd.read_csv(path+record) 
            name = record.split('.')[0] 
            ## Full signal segmentation
            df_sp = df[[col for col in df.columns if col[:2]=='Sp']] 
            signal = df_sp.to_numpy()
     
            bkps=signal_segmentation(signal,
                                     nb_bkps,
                                     )
            if display:
                display_segmentation(signal, 
                                     bkps, 
                                     name)
            bkps.insert(0, 0)
            filename = '{out_}{name_}.npy'.format(out_=outpath, 
                                                  name_=name)
            np.save(filename, np.array(bkps))
        
        
def process_oculomotor(config, path, feature_records, 
                       display=False):
    '''
    

    Parameters
    ----------
    config : TYPE
        DESCRIPTION.
    path : TYPE
        DESCRIPTION.
    feature_records : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    nb_bkps = config['symbolization']['segmentation']['oculomotor']['nb_breakpoints']
    outpath = 'output/results/ETRA/segmentation/'
 
    for record in feature_records: 
        subject, trial, task, condition, stimulus, _ = record.split('.')[0].split('_')
        
        if (subject in config['data']['subject_set']
            and task in config['data']['task_set']
            and condition in config['data']['condition_set']):
            
            df = pd.read_csv(path+record) 
            name = record.split('.')[0]
     
            ## Full signal segmentation
            #signal = df.to_numpy()[:,1:]
            #bkps=signal_segmentation(signal,
            #                         nb_bkps,
            #                         )
            #if display:
            #    display_segmentation(signal, 
            #                         bkps, 
            #                         name)
            #bkps.insert(0, 0)
            #filename = '{out_}{name_}.npy'.format(out_=outpath, 
            #                                      name_=name)
            #np.save(filename, np.array(bkps))
            
            ## Segmentation of fixation features
            df_fix = df[[col for col in df.columns if col[:3]=='fix']] 
            signal_fix = df_fix.to_numpy()
            bkps=signal_segmentation(signal_fix,
                                     nb_bkps,
                                     )
            if display:
                display_segmentation(signal_fix, 
                                     bkps, 
                                     name+'_fixationFeatures')
            bkps.insert(0, 0)
            filename = '{out_}{name_}Fixation.npy'.format(out_=outpath, 
                                                          name_=name)
            np.save(filename, np.array(bkps))
            
            
            ## Segmentation of saccade features
            df_sac = df[[col for col in df.columns if col[:3]=='sac']] 
            signal_sac = df_sac.to_numpy()
            bkps=signal_segmentation(signal_sac,
                                     nb_bkps,
                                     )
            if display:
                display_segmentation(signal_sac, 
                                     bkps, 
                                     name+'_saccadeFeatures')
            bkps.insert(0, 0)
            filename = '{out_}{name_}Saccade.npy'.format(out_=outpath, 
                                                         name_=name)
            np.save(filename, np.array(bkps))
        
        
    
def signal_segmentation(signal, 
                        nb_bkps = 4):
    '''
    

    Parameters
    ----------
    signal : TYPE
        DESCRIPTION.
    nb_bkps : TYPE, optional
        DESCRIPTION. The default is 4.
    pen : TYPE, optional
        DESCRIPTION. The default is .5.

    Returns
    -------
    my_bkps : TYPE
        DESCRIPTION.

    '''
   
    if nb_bkps is not None: 
        algo = rpt.KernelCPD(kernel="linear", jump=1).fit(signal)
        my_bkps = algo.predict(n_bkps=nb_bkps)
        
    else:
        pen = np.log(signal.shape[0])/10
        model='l2' # "l1", "rbf"
        algo = rpt.Pelt(model=model, jump=1).fit(signal)
        my_bkps = algo.predict(pen=pen)
    
    return my_bkps
    
 
def display_segmentation(signal, my_bkps, name=None):
    '''
    

    Parameters
    ----------
    signal : TYPE
        DESCRIPTION.
    my_bkps : TYPE
        DESCRIPTION.
    name : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''
    
    plt.style.use("seaborn-v0_8")  
    
    plt.plot(signal)
    for x in my_bkps[:-1]:
        plt.axvline(x = x-1, color = 'indianred', 
                    linewidth=5, linestyle='dashed')
    if name is not None:
        plt.title(name)
    plt.show()
    plt.clf()
  
    fig = plt.figure()
    ax = fig.add_subplot(111)  
    ax.imshow(signal.T, aspect=2.5, cmap='viridis', vmin=0, vmax=1)
    ax.grid(None)
    plt.show()
    plt.clf()
    
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    
    
    ax.imshow(signal.T, aspect=2.5, cmap='viridis', vmin=0, vmax=1)
    ax.grid(None)
    
    for x in my_bkps[:-1]:
        ax.axvline(x = x-.5, color = 'indianred', 
                    linewidth=5, linestyle='dashed')
    #if name is not None:
    #    ax.title(name)
         
   
    plt.show()
    plt.clf()
    
    
    
 





