# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, squareform
from fastcluster import linkage



class Symbolization():
    
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
        if True:
            ## Process symbolization for fixation features
            oculomotor_features = self.config['data']['oculomotor_features'] 
            oculomotor_feature_records = [feature_record for feature_record in self.feature_records
                                          if feature_record.split('.')[0].split('_')[-1] == 'oculomotor']
            fix_feature_set = [feature for feature in oculomotor_features if feature[:3]=='fix']
            self.process_subset(oculomotor_feature_records,
                                fix_feature_set, 
                                'oculomotorFixation') 
        if True:
            ## Process symbolization for saccade features
            oculomotor_features = self.config['data']['oculomotor_features'] 
            oculomotor_feature_records = [feature_record for feature_record in self.feature_records
                                          if feature_record.split('.')[0].split('_')[-1] == 'oculomotor']
            sac_feature_set = [feature for feature in oculomotor_features if feature[:3]=='sac']
            self.process_subset(oculomotor_feature_records,
                                sac_feature_set, 
                                'oculomotorSaccade')
        if True:
            ## Process symbolization for scanpath features
            scanpath_features = self.config['data']['scanpath_features'] 
            scanpath_feature_records = [feature_record for feature_record in self.feature_records
                                        if feature_record.split('.')[0].split('_')[-1] == 'scanpath']
            sp_feature_set = [feature for feature in scanpath_features if feature[:2]=='Sp']
            self.process_subset(scanpath_feature_records,
                                sp_feature_set, 
                                'scanpath')
        if True:
            ## Process symbolization for aoi sequence features 
            aoi_features = self.config['data']['aoi_features']
            aoi_features_records = [feature_record for feature_record in self.feature_records
                                    if feature_record.split('.')[0].split('_')[-1] == 'AoI']
            aoi_feature_set = [feature for feature in aoi_features if feature[:3] == 'AoI']
            self.process_subset(aoi_features_records, 
                                aoi_feature_set, 
                                'AoI')




    def process_subset(self, 
                       feature_records, feature_set, type_, 
                       display=True):
        '''
        

        Parameters
        ----------
        feature_records : TYPE
            DESCRIPTION.
        feature_set : TYPE
            DESCRIPTION.
        type_ : TYPE
            DESCRIPTION.
        display : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        '''
        bkpt_path = 'output/ADABase/segmentation/'
        if type_=='scanpath':
            n_centers = self.config['symbolization']['nb_clusters']['scanpath']     
        elif type_=='AoI':
            n_centers = self.config['symbolization']['nb_clusters']['aoi'] 
        else:
            n_centers = self.config['symbolization']['nb_clusters']['oculomotor'] 
        
        if self.config['symbolization']['normalization'] == 'all': 
            ## Initialize concatenated data for all subjects
            sub_data = []
            for record in feature_records:  
                subject, study, phase, level, _ = record.split('.')[0].split('_')
                label = '_'.join([study, phase, level])
                if subject in self.config['data']['subject_set'] and label in self.config['data']['label_set']:
                    df = pd.read_csv(self.path+record)[feature_set]  
                     
                    df = df.to_numpy()  
                    name = record.split('.')[0]  
                    bkpt_name = '{s_}_{l_}_{type_}.npy'.format(s_=subject,
                                                               l_=label,
                                                               type_=type_)
                   
                    bkpts = np.load(bkpt_path+bkpt_name, allow_pickle=True ) 
                    for i in range(1, len(bkpts)):
                        l_data = df[bkpts[i-1]: bkpts[i]] 
                        l_means = np.mean(l_data, axis=0) 
                        sub_data.append(l_means)
            sub_data = np.array(sub_data)
            kmeans = KMeans(n_clusters=n_centers, 
                            n_init=100, 
                            random_state=0).fit(sub_data)
            centers = kmeans.cluster_centers_ 
            dist_mat = cdist(centers, centers)
        
            ## Re-order clusters according to pairwise distances 
            ordered_dist_mat, res_order, res_linkage = self.compute_serial_matrix(dist_mat, 
                                                                                  'ward')
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
            ## Second pass
            for record in feature_records:   
                subject, study, phase, level, _ = record.split('.')[0].split('_')
                label = '_'.join([study, phase, level])
                if subject in self.config['data']['subject_set'] and label in self.config['data']['label_set']:
                    sub_data = []
                    lengths = []
                    df = pd.read_csv(self.path+record)[feature_set]  
                    df = df.to_numpy() 
                    name = record.split('.')[0] 
                    bkpt_name = '{s_}_{l_}_{type_}.npy'.format(s_=subject,
                                                               l_=label,
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
                    name = '{s_}_{l_}'.format(s_=subject, 
                                              l_=label)
  
                    result_dict['recordings'].update({name: dict()})
                    result_dict['recordings'][name].update({'sequence': ordered_labs_, 
                                                            'lengths': lengths}) 
            filename = '{outpath}/{type_}.pkl'.format(outpath='output/ADABase/symbolization',  
                                                      type_=type_)
            with open(filename, 'wb') as fp:
                pickle.dump(result_dict, fp)   


    
    def seriation(self, 
                  Z,N,cur_index):
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
            return (self.seriation(Z,N,left) + self.seriation(Z,N,right))
        
        
    def compute_serial_matrix(self, 
                              dist_mat,
                              method="ward"):
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
        res_order = self.seriation(res_linkage, N, N + N-2)
        
        seriated_dist = np.zeros((N,N))
        a,b = np.triu_indices(N,k=1)
        
        seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
        seriated_dist[b,a] = seriated_dist[a,b]
      
        return seriated_dist, res_order, res_linkage   
    
    
    
    





































