# -*- coding: utf-8 -*-

import yaml
import os 
import pandas as pd
import processing.ADABase.block_analysis as ba
import processing.ADABase.normalization as nm
import processing.ADABase.segmentation as sg
import processing.ADABase.symbolization as sy
import processing.ADABase.clustering as cl

import matplotlib.pyplot as plt   

def feature_extraction(config=None):
    '''
    

    Returns
    -------
    None.

    '''
    if config is None:
        with open('configurations/analysis_adabase.yaml', 'r') as file:
            config = yaml.safe_load(file)
     
    path = 'dataset/ADABAse/parsed_data/'
    records = [f for f in os.listdir(path) if f[-4:] == '.csv']
    
    block_a = ba.BlockAnalysis(config, path, records) 
    block_a.process()
            
           
def feature_normalization(config=None):
    '''
    

    Returns
    -------
    None.

    '''
    if config is None:
        with open('configurations/analysis_adabase.yaml', 'r') as file:
            config = yaml.safe_load(file)
            
    path = 'output/ADABAse/features/'
    feature_records = [f for f in os.listdir(path) if f[-4:] == '.csv']

    norm_a = nm.Normalization(config, path, feature_records)
    norm_a.process()
    
    
def segmentation(config=None):
    '''
    

    Returns
    -------
    None.

    '''
    if config is None:
        with open('configurations/analysis_adabase.yaml', 'r') as file:
            config = yaml.safe_load(file) 
        
    path = 'output/ADABAse/normalized_features/'
    feature_records = [f for f in os.listdir(path) if f[-4:] == '.csv']

    seg_a = sg.Segmentation(config, path, feature_records)
    seg_a.process()


def symbolization(config=None):
    '''
    

    Returns
    -------
    None.

    '''
    if config is None:
        with open('configurations/analysis_adabase.yaml', 'r') as file:
            config = yaml.safe_load(file) 
            
    path = 'output/results/ADABAse/normalized_features/'
    feature_records = [f for f in os.listdir(path) if f[-4:] == '.csv']

    symb_a = sy.Symbolization(config, path, feature_records) 
    symb_a.process()
    
    
def clustering(config=None):
    '''
     

    Returns
    -------
    None.

    '''
    if config is None:
        with open('configurations/analysis_adabase.yaml', 'r') as file:
            config = yaml.safe_load(file) 
            
    path = 'output/ADABase/symbolization/'
    symb_results = [f for f in os.listdir(path) if f[-4:] == '.pkl']
   
    clu_a = cl.Clustering(config, path, symb_results)
    clu_a.process()
     
    
    
if __name__ == '__main__': 
  
 
    if True:
        if False:                 
            feature_extraction() 
        if False:
            feature_normalization() 
        if True:
            segmentation() 
        if False:
            symbolization()
        if False:
            clustering()
                        
            
            
             
            