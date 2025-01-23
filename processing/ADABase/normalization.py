# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
 


class Normalization():
    
    def __init__(self, 
                 config, path, records):
        '''
        

        Parameters
        ----------
        config : TYPE
            DESCRIPTION.
        path : TYPE
            DESCRIPTION.
        records : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.config = config
        self.path = path 
        self.feature_records = records 
        
         
    def process(self): 
        '''
        

        Returns
        -------
        None.

        '''
        ## To avoid analysis with few available segments
        to_keep = []
        thr = self.config['general']['available_segment_min']
        for subject in self.config['data']['subject_set']: 
            for label in self.config['data']['label_set']:   
                df_o = pd.read_csv(self.path + '{s}_{l}_oculomotor.csv'.format(s=subject, 
                                                                               l=label))
                l_o = len(df_o)
                df_s = pd.read_csv(self.path + '{s}_{l}_scanpath.csv'.format(s=subject, 
                                                                             l=label))
                l_s = len(df_s)
                df_a = pd.read_csv(self.path + '{s}_{l}_AoI.csv'.format(s=subject, 
                                                                        l=label))
                l_a = len(df_a)
                if l_o >= thr and l_s >= thr and l_a >= thr:
                    to_keep.append('{s}_{l}'.format(s=subject, 
                                                    l=label)) 
        if True:
            oculomotor_feature_records = [r + '_oculomotor.csv' for r in to_keep] 
            self.process_normalization(oculomotor_feature_records, 
                                       type_='oculomotor')
        if True:
            scanpath_feature_records = [r + '_scanpath.csv' for r in to_keep] 
            self.process_normalization(scanpath_feature_records, 
                                       type_='scanpath')
        if True:
            aoi_feature_records = [r + '_AoI.csv' for r in to_keep] 
            self.process_normalization(aoi_feature_records, 
                                       type_='AoI')
       
            
    def process_normalization(self, 
                              feature_records, type_):
        '''
        

        Parameters
        ----------
        feature_records : TYPE
            DESCRIPTION.
        type_ : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        ## Dict of methods for normalization
        dict_methods=dict({'empirical': self.empirical_cdf}) 
        ## Compute data dict and parameter dict 
        data = dict()
        dict_norm = dict()
        
        if type_=='oculomotor':
            features = self.config['data']['oculomotor_features']
        elif type_=='scanpath':
            features = self.config['data']['scanpath_features']
        elif type_=='AoI':
            features = self.config['data']['aoi_features']
            
        for record in feature_records:   
            subject, study, phase, level, _ = record.split('.')[0].split('_')
            label = '_'.join([study, phase, level])
            
            if (subject in self.config['data']['subject_set'] and label in self.config['data']['label_set']):   
                df = pd.read_csv(self.path+record)  
                df=df.interpolate(axis=0).ffill().bfill() 
                data.update({record.split('.')[0]: df})
        ## Get CdF for each subject and each feature 
        if self.config['symbolization']['normalization'] == 'longitudinal':
            for subject in self.config['data']['subject_set']: 
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
                        feat_params = dict_methods[self.config['symbolization']['normalization_method']](ts, name)
                        dict_norm[subject].update({feature: feat_params})
        ## Get CdF for each feature       
        elif self.config['symbolization']['normalization'] == 'all':
            for feature in features:  
                if feature != 'startTime(s)':  
                    ## Concatenate data for all subjects
                    ts = []
                    for file in data.keys(): 
                        l_data = data[file]
                        ts += list(l_data[feature].values)  
                    name=feature  
                    feat_params = dict_methods[self.config['symbolization']['normalization_method']](ts, name)
                    dict_norm.update({feature: feat_params}) 
        ## Re-iterate to normalize according to n_params
        for file in data.keys():  
            subject, study, phase, level, _ = file.split('.')[0].split('_')
            label = '_'.join([study, phase, level])
            
            l_data = data[file]
            new_data = dict({})
            
            for feature in features: 
                ts = l_data[feature].values
                if feature != 'startTime(s)':
                    if self.config['symbolization']['normalization'] == 'longitudinal': 
                        if self.config['symbolization']['normalization_method'] == 'empirical':
                            ecdf = dict_norm[subject][feature]
                            ## To uniform distribution 
                            ts_n = ecdf.evaluate(ts)
                            new_data.update({feature: ts_n})
                    elif self.config['symbolization']['normalization'] == 'all':
                        if self.config['symbolization']['normalization_method'] == 'empirical':
                            ecdf = dict_norm[feature]
                            ## To uniform distribution 
                            ts_n = ecdf.evaluate(ts)
                            new_data.update({feature: ts_n})
                else:
                    new_data.update({feature: ts})
            new_df = pd.DataFrame.from_dict(new_data)    
            filename = 'output/results/ADABase/normalized_features/{f_}.csv'.format(f_=file) 
            new_df.to_csv(filename, index=False)
            
             
    def empirical_cdf(self, 
                      time_series, 
                      name=None, display=True):
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
                path= 'output/results/ADABase/figures/normalization/'
                #fig.savefig(path+name) 
            plt.show() 
            plt.clf()
            
            #plt.hist(ecdf.evaluate(time_series), bins=50,alpha=.3, density=True)
            #if name is not None:
            #    plt.title(name.split('_')[-1])
            #plt.show() 
            #plt.clf()
      
        return ecdf        
                
            
            
            
            
            
            
            