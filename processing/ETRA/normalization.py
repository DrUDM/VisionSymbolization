# -*- coding: utf-8 -*-

 
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
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
    if True:
        oculomotor_feature_records = [r for r in feature_records if r.split('.')[0][-10:]=='oculomotor']
        process_normalization(config, path, 
                              oculomotor_feature_records, 
                              type_='oculomotor')
    if True:
        scanpath_feature_records = [r for r in feature_records if r.split('.')[0][-8:]=='scanpath']
        process_normalization(config, path, 
                               scanpath_feature_records, 
                               type_='scanpath')
    if True:
        aoi_feature_records = [r for r in feature_records if r.split('.')[0][-3:]=='AoI']
        process_normalization(config, path, 
                              aoi_feature_records, 
                              type_='AoI')
    
def process_normalization(config, path, 
                          feature_records, 
                          type_):
    '''
    

    Parameters
    ----------
    config : TYPE
        DESCRIPTION.
    path : TYPE
        DESCRIPTION.
    feature_records : TYPE
        DESCRIPTION.
    type_ : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    ## Dict of methods for normalization
    dict_methods=dict({'log_normal': lognormal_uniformization, 
                       'empirical': empirical_cdf})
     
    ## Compute data dict and parameter dict 
    data = dict()
    dict_norm = dict()
    if type_=='oculomotor':
        features = config['data']['oculomotor_features']
    elif type_=='scanpath':
        features = config['data']['scanpath_features']
    elif type_=='AoI':
        features = config['data']['aoi_features']
    
    for record in feature_records:  
        subject, trial, task, condition, stimulus, _ = record.split('.')[0].split('_')
        
        if (subject in config['data']['subject_set']
            and task in config['data']['task_set']
            and condition in config['data']['condition_set']):
            
            df = pd.read_csv(path+record)  
            df=df.interpolate(axis=0).ffill().bfill() 
            data.update({record.split('.')[0]: df})
   
    if config['symbolization']['normalization'] == 'longitudinal':
        #print('\nProcessing longitudinal normalization for {type_} features'.format(type_=type_))
        for subject in config['data']['subject_set']: 
            dict_norm[subject] = dict() 
            for feature in features:  
                if feature != 'startTime(s)':  
                    ## Concatenate data for a same subject
                    ts = []
                    for file in data.keys(): 
                        if file.split('_')[0] == subject: 
                            l_data = data[file]
                            ts += list(l_data[feature].values)   
                    name=record.split('.')[0]+'_'+feature  
                    feat_params = dict_methods[config['symbolization']['normalization_method']](ts, name)
                    dict_norm[subject].update({feature: feat_params})
 
    elif config['symbolization']['normalization'] == 'all':
        #print('\nProcessing normalization for {type_} features'.format(type_=type_))
        for feature in features:  
            if feature != 'startTime(s)':  
                ## Concatenate data for all subjects
                ts = []
                for file in data.keys(): 
                    l_data = data[file]
                    ts += list(l_data[feature].values)  
                name=record.split('.')[0]+'_'+feature  
                feat_params = dict_methods[config['symbolization']['normalization_method']](ts, name)
                dict_norm.update({feature: feat_params})
        
    ## Re-iterate to normalize according to n_params
    for file in data.keys(): 
        print('\nAnalyzing file {rec_}'.format(rec_=file))
        subject, trial, task, condition, stimulus, _ = file.split('.')[0].split('_')
        l_data = data[file]
        new_data = dict({})
        
        for feature in features: 
            ts = l_data[feature].values
            if feature != 'startTime(s)':
                if config['symbolization']['normalization'] == 'longitudinal': 
                    if config['symbolization']['normalization_method'] == 'log_normal':
                        param = dict_norm[subject][feature]
                        ## To uniform distribution 
                        ts_n = sp.stats.lognorm.cdf(ts, 
                                                    param[0], 
                                                    loc=param[1], 
                                                    scale=param[2])
                        new_data.update({feature: ts_n})
                        
                    elif config['symbolization']['normalization_method'] == 'empirical':
                        ecdf = dict_norm[subject][feature]
                        ## To uniform distribution 
                        ts_n = ecdf.evaluate(ts)
                        new_data.update({feature: ts_n})
                        
                elif config['symbolization']['normalization'] == 'all':
                    if config['symbolization']['normalization_method'] == 'log_normal':
                        param = dict_norm[feature]
                        ## To uniform distribution 
                        ts_n = sp.stats.lognorm.cdf(ts, 
                                                    param[0], 
                                                    loc=param[1], 
                                                    scale=param[2])
                        new_data.update({feature: ts_n})
                        
                    elif config['symbolization']['normalization_method'] == 'empirical':
                        ecdf = dict_norm[feature]
                        ## To uniform distribution 
                        ts_n = ecdf.evaluate(ts)
                        new_data.update({feature: ts_n})
            else:
                new_data.update({feature: ts})
                
        new_df = pd.DataFrame.from_dict(new_data)    
        filename = 'output/ETRA/normalized_features/{f_}.csv'.format(f_=file) 
        new_df.to_csv(filename, index=False)
                
                
def lognormal_uniformization(time_series, 
                             name=None):
    '''
    

    Parameters
    ----------
    time_series : TYPE
        DESCRIPTION.
    name : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    param : TYPE
        DESCRIPTION.

    ''' 
    param=sp.stats.lognorm.fit(time_series)
   
    x=np.linspace(0,max(time_series),250)
    pdf_fitted = sp.stats.lognorm.pdf(x, param[0], loc=param[1], scale=param[2])
    try:
        plt.style.use("seaborn-v0_8") 
        plt.hist(time_series,bins=50,alpha=.3, density=True) 
        plt.plot(x,pdf_fitted,'r-')
        
        if name is not None:
            plt.title(name.split('_')[-1])
            fig = plt.gcf()
            path= 'output/ETRA/figures/normalization/'
            fig.savefig(path+name)
            
        plt.show() 
        plt.clf()
        
    except:
        pass
     
    return param
    

def empirical_cdf(time_series, 
                  name=None, 
                  display=False):
    '''
    

    Parameters
    ----------
    time_series : TYPE
        DESCRIPTION.
    name : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    ecdf : TYPE
        DESCRIPTION.

    ''' 
    res = stats.ecdf(time_series)
    ecdf = res.cdf
 
    if display:
        plt.style.use("seaborn-v0_8") 
        plt.hist(time_series,bins=50,alpha=.3, density=True) 
         
        if name is not None:
            plt.title(name.split('_')[-1])
            fig = plt.gcf()
            path= 'output/ETRA/figures/normalization/'
            fig.savefig(path+name)
            
        plt.show() 
        plt.clf()
 
      
    return ecdf
    
    
    