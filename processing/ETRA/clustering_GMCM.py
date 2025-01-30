# -*- coding: utf-8 -*-

import numpy as np
import pickle
import matplotlib.pyplot as plt   
from scipy.spatial.distance import squareform
from fastcluster import linkage
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.manifold import MDS

from weighted_levenshtein import lev  
from processing.c_comparison_algorithms import c_comparison_algorithms as c_comparison



class ClusteringGMCM():
    
    def __init__(self, 
                 config, path, 
                 symbolization_results):
        '''
        

        Parameters
        ----------
        config : TYPE
            DESCRIPTION.
        path : TYPE
            DESCRIPTION.
        symbolization_results : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.config = config
        self.path = path 
        self.symbolization_results = symbolization_results


    def process(self):
        '''
        

        Returns
        -------
        None.

        '''
        if self.config['symbolization']['normalization'] == 'longitudinal':
            self.process_l()
            
        elif self.config['symbolization']['normalization'] == 'all':
            if self.config['clustering']['method'] == 'svm':
                self.process_a_svm_dist() 
          
  
    def process_a_svm_dist(self):
        '''
        

        Returns
        -------
        None.

        '''
        to_cluster = 'tasks' #'subjects
        
        conditions = sorted(self.config['data']['condition_set'])
        subject_set = sorted(self.config['data']['subject_set'])
        conditions_dict = dict({})
        subject_dict = dict()
        
        for i, cond_ in enumerate(conditions):
            conditions_dict.update({cond_: i})
        for i, sub_ in enumerate(subject_set):
            subject_dict.update({sub_: i})
            
        binning = self.config['symbolization']['binning'] 
        symb = [f for f in self.symbolization_results if f.split('.')[0] == 'AoI'][0]
        with open(self.path+symb, 'rb') as f:
            symb = pickle.load(f)
        records = sorted(list(symb['recordings'].keys())) 
        records = [r_ for r_ in records if r_.split('_')[3] in conditions]
         
        y_ = []
        for record in records: 
            if to_cluster == 'tasks':
                y_.append(record.split('_')[3])
            elif to_cluster == 'subjects':
                y_.append(record.split('_')[0])
      
        records = np.array(records)
        y_ = np.array(y_)
    
        #y_[y_=='Blank']='Natural'
        #y_[y_=='Waldo']='Puzzle'
        
        dist_dict = dict({})
        for type_ in ['oculomotorFixation',
                      'oculomotorSaccade', 
                      'scanpath', 
                      'AoI'
                      ]:
            print('Processing {type_} distances...'.format(type_=type_))
            
            symb = [f for f in self.symbolization_results if f.split('.')[0] == type_][0]
            with open(self.path+symb, 'rb') as f:
                symb = pickle.load(f)
        
            ## Get distance matrix and normalize it
            d_m = symb['distance_matrix']
            d_m = (d_m)/np.max(d_m)  
            
            nb_c = d_m.shape[0]
            i_dict = dict()
            for i in range(nb_c):
                i_dict.update({chr(i + 65): i})
         
            record_dict = dict({})
            record_dict_idx = dict({})
            
            for i, record in enumerate(records): 
                seq = symb['recordings'][record]['sequence']
                l_ = symb['recordings'][record]['lengths'] 
                seq_=[]
               
                if binning:
                    for g in range(len(seq)):  
                        [seq_.append(chr(seq[g] + 65)) for _ in range(l_[g])]
                else:  
                    [seq_.append(chr(seq[g] + 65)) for g in range(len(seq))]
                record_dict.update({record: seq_})
                record_dict_idx.update({record: i})
          
            ## Compute distance matrix for a given modality
           
            dist_mat = np.zeros((len(records), len(records))) 
            for j in range(1, len(records)): 
                for i in range(j):     
                    s_1 = record_dict[records[i]]
                    s_2 = record_dict[records[j]] 
                    ed = GeneralizedEditDistance(s_1, s_2, 
                                                 d_m, i_dict, 
                                                 self.config)
                    ed.process()
                    d_ = ed.dist_
                    dist_mat[i,j]=dist_mat[j,i]=d_
         
            dist_dict.update({type_: dist_mat})
        ## Compute the 'final' disatnce matrix
        t_dist=np.zeros((len(records), len(records)))
        for k_ in dist_dict.keys(): 
            t_dist += dist_dict[k_]**1
        #t_dist=np.sqrt(t_dist)
        
        ordered_t_dist, _, _ = compute_serial_matrix(t_dist, 'ward')
        plt.imshow(ordered_t_dist)
        plt.grid(None) 
        plt.show()
        plt.clf()
        ## Embedding in an enclidean metric space from distance matrix
        print('Computing embedding...')
        embedding = MDS(n_components=120, 
                        dissimilarity='precomputed', 
                        normalized_stress='auto', 
                        random_state=1)
        X_embed = embedding.fit_transform(t_dist)
       
        print('Computing classification...')
        m_acc = []
        best, best_s, best_confmat = 0, None, None 
        
        for state in range(0, 1000): 
            accuracies = []
            kf = KFold(n_splits=5, random_state=state, shuffle=True)
            
            if to_cluster == 'tasks':
                conf_mat = np.zeros((len(conditions), len(conditions)))
            elif to_cluster == 'subjects':
                conf_mat = np.zeros((len(subject_set), len(subject_set)))
                
            for i, (train_index, test_index) in enumerate(kf.split(records)): 
                X_train_r = records[train_index]
                X_test_r = records[test_index]
                
                y_train = y_[train_index]
                y_test = y_[test_index]
                
                X_train = np.array([X_embed[record_dict_idx[x_train_r]] for x_train_r in X_train_r])
                X_test = np.array([X_embed[record_dict_idx[x_test_r]] for x_test_r in X_test_r])
                 
                #clf = OneVsRestClassifier(SVC(C=1, kernel='rbf'))
                clf = SVC(C=1, kernel='rbf',)# gamma=0.5e0)
                clf.fit(X_train, y_train) 
                y_pred = clf.predict(X_test)
                #print(clf.get_params())
                correct=0
                for i in range(len(y_pred)): 
                    true_lab = y_test[i]
                    exp_lab = y_pred[i]
                    
                    if to_cluster == 'tasks':
                        conf_mat[conditions_dict[true_lab], 
                                 conditions_dict[exp_lab]] +=1
                    elif to_cluster == 'subjects':
                        conf_mat[subject_dict[true_lab], 
                                 subject_dict[exp_lab]] +=1
                    
                    if true_lab==exp_lab:
                        correct+=1
                    #else:
                    #    print(true_lab)
                    #    print(exp_lab) 
                    #    print()
                        
                accuracy=correct/len(y_pred)
                #print('Accuracy: ' + str(accuracy)) 
                accuracies.append(accuracy)
            
            print('Mean accuracy: ' + str(np.mean(accuracies)) + ', for state ' + str(state)) 
          
            m_acc.append(np.mean(accuracies))
            if np.mean(accuracies) > best:
                best = np.mean(accuracies)
                best_s = state
                best_confmat = conf_mat
        print('Final accuracy: ' + str(best) + ', for state ' + str(best_s))
        print('Confusion matrix:')
        print(best_confmat)
        #print(np.mean(m_acc)) 
        
        disp=ConfusionMatrixDisplay(best_confmat, 
                                    display_labels=clf.classes_)
        disp.plot(values_format='', 
                  colorbar=False, 
                  cmap='Blues') 
        plt.show() 
        disp.figure_.savefig('output/ETRA/clustering/copula_svm_accuracy_{n}.png'.format(n=len(conditions)), 
                             dpi=250)
        plt.clf()
    
 

        
class GeneralizedEditDistance():
    
    def __init__(self, 
                 s_1, s_2, 
                 d_m, i_dict,
                 config):
        '''
        

        Parameters
        ----------
        s_1 : TYPE
            DESCRIPTION.
        s_2 : TYPE
            DESCRIPTION.
        centers : TYPE
            DESCRIPTION.
        config : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
         
        self.s_1, self.s_2 = s_1, s_2
        self.n_1 = len(s_1)
        self.n_2 = len(s_2)
        
        self.d_m = d_m
        self.i_dict = i_dict
        
        self.c_del = config['clustering']['edit_distance']['deletion_cost']
        self.c_ins = config['clustering']['edit_distance']['insertion_cost'] 
        self.norm_ = config['clustering']['edit_distance']['normalization']
        
        
    def process(self, 
                custom=False):
        '''
        

        Parameters
        ----------
        custom : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        '''
        d_m, i_dict = self.d_m, self.i_dict
        
        if custom:
            dist_ = c_comparison.generalized_edit(self.s_1, self.s_2,
                                                  self.c_del, self.c_ins,
                                                  i_dict, d_m)
            if self.norm_ == 'max':        
                self.dist_ = dist_ / max(self.n_1, self.n_2) 
            else:        
                self.dist_ = dist_ / min(self.n_1, self.n_2)
        
        else:
            substitute_costs = np.ones((128, 128), dtype=np.float64) 
            d_ = d_m.shape[0]
            substitute_costs[65: 65+d_, 
                             65: 65+d_] = d_m
            insert_costs = np.ones(128, dtype=np.float64)
            delete_costs = np.ones(128, dtype=np.float64)
            
            s_1 = ''.join(self.s_1)
            s_2 = ''.join(self.s_2)
            dist_ = lev(s_1, s_2, 
                        insert_costs=insert_costs, 
                        delete_costs=delete_costs, 
                        substitute_costs=substitute_costs)
            self.dist_ = dist_ / max(self.n_1, self.n_2) 
     
                
           
             
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
            
                 



