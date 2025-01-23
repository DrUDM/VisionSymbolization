# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 

from sklearn.cluster import KMeans, SpectralClustering 
from scipy.spatial.distance import cdist, squareform
from fastcluster import linkage


def process(config, path, 
            feature_records):
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
        scanpath_features = config['data']['scanpath_features'] 
        scanpath_feature_records = [feature_record for feature_record in feature_records
                                    if feature_record.split('.')[0].split('_')[-1] == 'scanpath']
        sp_feature_set = [feature for feature in scanpath_features if feature[:2]=='Sp']
        process_subset(config, path, 
                       scanpath_feature_records,
                       sp_feature_set, 
                       'scanpath')
    if True:
        ## Process symbolization for aoi sequence features 
        aoi_features = config['data']['aoi_features']
        aoi_features_records = [feature_record for feature_record in feature_records
                                if feature_record.split('.')[0].split('_')[-1] == 'AoI']
        aoi_feature_set = [feature for feature in aoi_features if feature[:3] == 'AoI']
        process_subset(config, path, 
                       aoi_features_records, 
                       aoi_feature_set, 
                       'AoI')
    
            
def process_subset(config, path, 
                   feature_records, 
                   feature_set, 
                   type_, 
                   display=True):
    '''
    

    Parameters
    ----------
    config : TYPE
        DESCRIPTION.
    path : TYPE
        DESCRIPTION.
    feature_records : TYPE
        DESCRIPTION.
    feature_set : TYPE
        DESCRIPTION.
    type_ : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    ''' 
    bkpt_path = 'output/ETRA/segmentation/'
 
    if type_=='scanpath':
        n_centers = config['symbolization']['nb_clusters']['scanpath']     
    elif type_=='AoI':
        n_centers = config['symbolization']['nb_clusters']['aoi'] 
    else:
        n_centers = config['symbolization']['nb_clusters']['oculomotor'] 
    
    if config['symbolization']['normalization'] == 'longitudinal':
        for subject in config['data']['subject_set']: 
            ## Initialize concatenated data for each subject
            sub_data = []
            to_process = [record for record in feature_records if record.split('.')[0].split('_')[0]==subject]
            for record in to_process:  
                ## For each subject get feature set  
                df = pd.read_csv(path+record)[feature_set]  
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
            clus_ = KMeans(n_clusters=n_centers, 
                            n_init=100, 
                            random_state=0).fit(sub_data)
            #clus_ = SpectralClustering(n_clusters=n_centers).fit(sub_data)
            centers = clus_.cluster_centers_ 
            dist_mat = cdist(centers, centers)
        
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
            ordered_centers = []
            for k_ in range(len(centers)):
                ordered_centers.append(centers[int(res_order[k_])])
       
            rez=[]
            result_dict = dict({})
            result_dict.update({'centers': np.array(ordered_centers), 
                                'recordings': dict({})})
            
            for record in to_process:   
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
                labs_ = clus_.predict(sub_data)
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
                for lab in ordered_labs_:
                    rez.append(lab)
            #print(rez) 
            #for i in range(n_centers):
            #    print(rez.count(i))
            filename = '{outpath}/{subject}_{type_}.pkl'.format(outpath='output/ETRA/symbolization', 
                                                                subject=subject,
                                                                type_=type_)
            with open(filename, 'wb') as fp:
                pickle.dump(result_dict, fp)
           
             
    elif config['symbolization']['normalization'] == 'all': 
        ## Initialize concatenated data for all subjects
        sub_data = []
        for record in feature_records:  
            ## For each subject get feature set  
            df = pd.read_csv(path+record)[feature_set]  
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
        kmeans = KMeans(n_clusters=n_centers, 
                        n_init=100, 
                        random_state=0).fit(sub_data)
        centers = kmeans.cluster_centers_ 
        dist_mat = cdist(centers, centers)
    
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
        ordered_centers = []
        for k_ in range(len(centers)):
            ordered_centers.append(centers[int(res_order[k_])])
   
        rez=[]
        result_dict = dict({})
        result_dict.update({'centers': np.array(ordered_centers), 
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
            labs_ = kmeans.predict(sub_data)
            ordered_labs_ = re_ordering(labs_) 
            lengths = list(np.array(lengths)) 
            name = '{sub_}_{tri_}_{tas_}_{con_}_{sti_}'.format(sub_=subject_,
                                                               tri_=trial,
                                                               tas_=task,
                                                               con_=condition,
                                                               sti_=stimulus) 
            result_dict['recordings'].update({name: dict()})
            result_dict['recordings'][name].update({'sequence': ordered_labs_, 
                                                    'lengths': lengths}) 
            for lab in ordered_labs_:
                rez.append(lab)
        #print(rez) 
        #for i in range(n_centers):
        #    print(rez.count(i))
        filename = '{outpath}/{type_}.pkl'.format(outpath='output/ETRA/symbolization',  
                                                  type_=type_)
        with open(filename, 'wb') as fp:
            pickle.dump(result_dict, fp)   
       
        
           
    #with open('output/ETRA/symbolization/009_AoI.pkl', 'rb') as f:
    #    data = pickle.load(f)
    #print(data)
             
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
            
            
            
            
            
            
            