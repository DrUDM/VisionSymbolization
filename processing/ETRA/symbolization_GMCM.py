# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 
import copy

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, squareform
from fastcluster import linkage

from processing.ETRA.gmcm_ad import AD_GMCM
from processing.ETRA.gmcm_pem import PEM_GMCM


def process(config, path, 
            feature_records):
 
    
    if True:
        ## Process symbolization for fixation features
        oculomotor_features = config['data']['oculomotor_features'] 
        oculomotor_feature_records = [feature_record for feature_record in feature_records
                                      if feature_record.split('.')[0].split('_')[-1] == 'oculomotor']
        fix_feature_set = [feature for feature in oculomotor_features if feature[:3]=='fix']
        process_subset(config, path, 
                       oculomotor_feature_records,
                       fix_feature_set, 
                       'oculomotorFixation')
    if True:
        ## Process symbolization for saccade features
        oculomotor_features = config['data']['oculomotor_features'] 
        oculomotor_feature_records = [feature_record for feature_record in feature_records
                                      if feature_record.split('.')[0].split('_')[-1] == 'oculomotor']
        sac_feature_set = [feature for feature in oculomotor_features if feature[:3]=='sac']
        process_subset(config, path, 
                       oculomotor_feature_records,
                       sac_feature_set, 
                       'oculomotorSaccade')
    if True:
        ## Process symbolization for scanpath features
        config_ = copy.deepcopy(config)
        scanpath_features = config['data']['scanpath_features'] 
        master_stepsize = config['symbolization']['gmcm']['master_stepsize']
        config_['symbolization']['gmcm']['master_stepsize'] = master_stepsize/3
        scanpath_feature_records = [feature_record for feature_record in feature_records
                                    if feature_record.split('.')[0].split('_')[-1] == 'scanpath']
        sp_feature_set = [feature for feature in scanpath_features if feature[:2]=='Sp']
        process_subset(config_, path, 
                       scanpath_feature_records,
                       sp_feature_set, 
                       'scanpath')
    if True:
        ## Process symbolization for aoi sequence features 
        config_ = copy.deepcopy(config)
        aoi_features = config['data']['aoi_features']
        master_stepsize = config['symbolization']['gmcm']['master_stepsize']
        config_['symbolization']['gmcm']['master_stepsize'] = master_stepsize/2
        aoi_features_records = [feature_record for feature_record in feature_records
                                if feature_record.split('.')[0].split('_')[-1] == 'AoI']
        aoi_feature_set = [feature for feature in aoi_features if feature[:3] == 'AoI']
        process_subset(config_, path, 
                       aoi_features_records, 
                       aoi_feature_set, 
                       'AoI')
        
        
def process_subset(config, path, 
                   feature_records, 
                   feature_set, 
                   type_, 
                   display=True):
     
        
    bkpt_path = 'output/results/ETRA/segmentation/'
 
    if type_=='scanpath':
        nb_centers = config['symbolization']['nb_clusters']['scanpath']     
    elif type_=='AoI':
        nb_centers = config['symbolization']['nb_clusters']['aoi'] 
    else:
        nb_centers = config['symbolization']['nb_clusters']['oculomotor'] 
    
    
    if config['symbolization']['normalization'] == 'longitudinal':
        return 
    
    
    elif config['symbolization']['normalization'] == 'all': 
        ## Initialize concatenated data for all subjects
        sub_data = []
        for record in feature_records:   
            df = pd.read_csv(path+record)[feature_set] 
            df=df.interpolate(axis=0).ffill().bfill() 
            df = df.to_numpy() 
            subject_, trial, task, condition, stimulus, _ = record.split('.')[0].split('_')
            
            bkpt_name = '{sub_}_{tri_}_{tas_}_{con_}_{sti_}_{type_}.npy'.format(sub_=subject_,
                                                                                tri_=trial,
                                                                                tas_=task,
                                                                                con_=condition,
                                                                                sti_=stimulus, 
                                                                                type_=type_)
            bkpts = np.load(bkpt_path+bkpt_name) 
            for i in range(1, len(bkpts)):
                l_data = df[bkpts[i-1]: bkpts[i]] 
                l_means = np.mean(l_data, axis=0) 
                sub_data.append(l_means)
         
        sub_data=np.array(sub_data)   
        gmcm = AD_GMCM(nb_centers,
                       config, 
                       uniformized=True)
        gmcm.fit(sub_data)   
        dist_mat = gmcm.distance_matrix()
        
        ## Re-order clusters according to pairwise distances 
        ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(dist_mat, 'ward')
        ## Compute inv_res_order to change cluster labels and center labels
        inv_res_order = np.zeros(len(res_order))
        for k_ in range(len(res_order)):
            inv_res_order[res_order[k_]] = int(k_)
            
        if display:
            plt.style.use("seaborn-v0_8")  
            plt.imshow(ordered_dist_mat)
            plt.grid(None)
            plt.title(type_)
            plt.show()
            plt.clf()
        
        re_ordering = lambda x: [int(inv_res_order[x[i]]) for i in range(len(x))]
        result_dict = dict({})
        result_dict.update({'distance_matrix': np.array(ordered_dist_mat), 
                            'recordings': dict({})})
        for record in feature_records:   
            sub_data = []
            lengths = []
            df = pd.read_csv(path+record)[feature_set]  
            df = df.to_numpy() 
            subject_, trial, task, condition, stimulus, _ = record.split('.')[0].split('_')
            
            #if condition=='Blank':
            bkpt_name = '{sub_}_{tri_}_{tas_}_{con_}_{sti_}_{type_}.npy'.format(sub_=subject_,
                                                                                tri_=trial,
                                                                                tas_=task,
                                                                                con_=condition,
                                                                                sti_=stimulus, 
                                                                                type_=type_)
            bkpts = np.load(bkpt_path+bkpt_name) 
            for i in range(1, len(bkpts)): 
                l_data = df[bkpts[i-1]: bkpts[i]] 
                l_means = np.mean(l_data, axis=0)  
                sub_data.append(l_means)
                lengths.append(bkpts[i]-bkpts[i-1])
              
            sub_data=np.array(sub_data)
            labs_ = gmcm.predict(sub_data, uniformized=True)
            ordered_labs_ = re_ordering(labs_) 
            lengths = list(np.array(lengths)) 
            name = '{sub_}_{tri_}_{tas_}_{con_}_{sti_}'.format(sub_=subject_,
                                                               tri_=trial,
                                                               tas_=task,
                                                               con_=condition,
                                                               sti_=stimulus)
            
            result_dict['recordings'].update({name: dict({})})
            result_dict['recordings'][name].update({'sequence': ordered_labs_, 
                                                    'lengths': lengths})
            
        filename = '{outpath}/{type_}.pkl'.format(outpath='output/results/ETRA/symbolization_GMCM',  
                                                  type_=type_)
        with open(filename, 'wb') as fp:
            pickle.dump(result_dict, fp)   
            
             
            
def seriation(Z,N,cur_index):
    '''
    

    Parameters
    ----------
    Z : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.
    cur_index : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    ## Computes the order implied by a hierarchical tree (dendrogram) 
    if cur_index < N:
        return [cur_index]
    
    else: 
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1]) 
        return (seriation(Z,N,left) + seriation(Z,N,right))
   
    
    
def compute_serial_matrix(dist_mat,method="ward"):
    '''
    

    Parameters
    ----------
    dist_mat : TYPE
        DESCRIPTION.
    method : TYPE, optional
        DESCRIPTION. The default is "ward".

    Returns
    -------
    seriated_dist : TYPE
        DESCRIPTION.
    res_order : TYPE
        DESCRIPTION.
    res_linkage : TYPE
        DESCRIPTION.

    '''
    ## Transformsa distance matrix into a sorted distance matrix according to 
    ## the order implied by the hierarchical tree 
    N = len(dist_mat)
    
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method,preserve_input=True)
    ## Compute re-ordering
    res_order = seriation(res_linkage, N, N + N-2)
    
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]
  
    return seriated_dist, res_order, res_linkage     
            
            
        
            
            
            
            
            
            