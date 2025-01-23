# -*- coding: utf-8 -*-

 
import yaml
import os 

import processing.ETRA.block_analysis as ba
import processing.ETRA.normalization as nm
import processing.ETRA.segmentation as sg
import processing.ETRA.symbolization_GMCM as gmcm_sy
from processing.ETRA.clustering_GMCM import ClusteringGMCM

  
def feature_extraction(config=None):
    '''
    

    Returns
    -------
    None.

    '''
    if config is None:
        with open('configurations/analysis_etra.yaml', 'r') as file:
            config = yaml.safe_load(file)
     
    path = 'dataset/ETRA/parsed_data/'
    records = [f for f in os.listdir(path) if f[-4:] == '.csv']
     
    ba.new_process(config, path, records) 
            
           
def feature_normalization(config=None):
    '''
    

    Returns
    -------
    None.

    '''
    if config is None:
        with open('configurations/analysis_etra.yaml', 'r') as file:
            config = yaml.safe_load(file)
            
    path = 'output/results/ETRA/features/'
    feature_records = [f for f in os.listdir(path) if f[-4:] == '.csv']

    nm.process(config, path, feature_records)
    
    
def segmentation(config=None):
    '''
    

    Returns
    -------
    None.

    '''
    if config is None:
        with open('configurations/analysis_etra.yaml', 'r') as file:
            config = yaml.safe_load(file) 
        
    path = 'output/results/ETRA/normalized_features/'
    feature_records = [f for f in os.listdir(path) if f[-4:] == '.csv']

    sg.process(config, path, feature_records)


def symbolization(config=None):
    '''
    

    Returns
    -------
    None.

    '''
    if config is None:
        with open('configurations/analysis_etra.yaml', 'r') as file:
            config = yaml.safe_load(file) 
            
    path = 'output/results/ETRA/normalized_features/'
    feature_records = [f for f in os.listdir(path) if f[-4:] == '.csv']

    gmcm_sy.process(config, path, feature_records)
    
    
def clustering(config=None):
    '''
     

    Returns
    -------
    None.

    '''
    if config is None:
        with open('configurations/analysis_etra.yaml', 'r') as file:
            config = yaml.safe_load(file) 
            
    path = 'output/results/ETRA/symbolization_GMCM/'
    symb_results = [f for f in os.listdir(path) if f[-4:] == '.pkl']
   
    cl = ClusteringGMCM(config, path, symb_results)
    cl.process()
     
    
    
if __name__ == '__main__': 
  
    if False:
        with open('configurations/analysis_etra.yaml', 'r') as file:
            config = yaml.safe_load(file) 
        for o in range(10, 11):
            for s in range(8, 16):
                for a in range(8, 16):
                    config['symbolization']['nb_clusters']['oculomotor'] = o
                    config['symbolization']['nb_clusters']['scanpath'] = s
                    config['symbolization']['nb_clusters']['aoi'] = a
                 
                    print('{o}, {s}, {a}'.format(o=o, 
                                                 s=s, 
                                                 a=a))
                    if False:                 
                        feature_extraction(config) 
                    if False:
                        feature_normalization(config) 
                    if False:
                        segmentation(config) 
                    if False:
                        symbolization(config)
                    if False:
                        clustering(config)
    if True:
        if False:                 
            feature_extraction() 
        if False:
            feature_normalization() 
        if False:
            segmentation() 
        if False:
            symbolization()
        if True:
            clustering()
                        
            
            
             
            
            
            
            
            
            
            
            
            
            
            
            
            
            