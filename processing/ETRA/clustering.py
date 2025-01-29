# -*- coding: utf-8 -*-

import numpy as np
import pickle
import kmedoids
import matplotlib.pyplot as plt   
from scipy.spatial.distance import squareform
from fastcluster import linkage
from sklearn.metrics import ConfusionMatrixDisplay
 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.manifold import MDS 

from weighted_levenshtein import lev 
from vision_toolkit.utils.binning import aoi_dict_dist_mat
from processing.c_comparison_algorithms import c_comparison_algorithms as c_comparison


class Clustering():
    
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
            elif self.config['clustering']['method'] == 'k_nn':
                self.process_a_knn()
            elif self.config['clustering']['method'] == 'adaboost':
                self.process_a_ada_dist()
            elif self.config['clustering']['method'] == 'random_forest':
                self.process_a_rfor_dist()


    def process_l(self):
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
        conditions = sorted(self.config['data']['condition_set'])
        conditions_dict = dict({})
        for i, cond_ in enumerate(conditions):
            conditions_dict.update({cond_: i})
            
        binning = self.config['symbolization']['binning'] 
        accuracies=[]
        for subject in self.config['data']['subject_set']: 
            to_process = [f for f in self.symbolization_results if f[:3] == subject]
          
            ## Load symbolization for each modality
            dist_dict=dict({})
            for type_ in ['oculomotorFixation',
                          'oculomotorSaccade', 
                          'scanpath', 
                          'AoI']:
                #print('Processing {type_}...'.format(type_=type_))
                symb = [f for f in to_process if f.split('.')[0].split('_')[1] == type_][0]
                with open(self.path+symb, 'rb') as f:
                    symb = pickle.load(f)
                 
                centers = symb['centers']
                centers_dict = dict({})
                records = sorted(list(symb['recordings'].keys()))
                record_dict = dict({})
                
                for record in records: 
                    seq = symb['recordings'][record]['sequence']
                    l_ = symb['recordings'][record]['lengths'] 
                    seq_=[]
                    if binning:
                        for i in range(len(seq)):  
                            [seq_.append(chr(seq[i] + 65)) for _ in range((l_[i]))]
                    else:
                        [seq_.append(chr(seq[i] + 65)) for i in range(len(seq))]
              
                    record_dict.update({record: seq_})
                    
                for i in range(len(centers)):
                    centers_dict.update({chr(i + 65): centers[i]})
                     
                d_m, i_dict = aoi_dict_dist_mat(centers_dict,
                                                normalize=True) 
                dist_mat = np.zeros((len(records), len(records))) 
                for j in range(1, len(records)): 
                    for i in range(j):     
                        s_1 = record_dict[records[i]]
                        s_2 = record_dict[records[j]] 
                        ed = GeneralizedEditDistance(s_1, s_2, d_m, i_dict, self.config)
                        ed.process()
                        d_ = ed.dist_
                        dist_mat[i,j]=dist_mat[j,i]=d_
                #print(dist_mat[0])
                #print(records)
                #for i in range(len(dist_mat[0])):
                #    print(records[i])
                #    print(dist_mat[5,i])
                dist_dict.update({type_: dist_mat})
            
            t_dist=np.zeros((len(records), len(records)))
            for k_ in dist_dict.keys(): 
                t_dist += dist_dict[k_]
            #print('Clustering...')   
            km = kmedoids.KMedoids(4, method='fasterpam', 
                                   random_state = 0)
            c = km.fit(t_dist)
             
            clusters_ = c.labels_ 
            #center_idx_ = c.medoid_indices_ 
            #for i, r_ in enumerate(records):
            #    print(r_)
            #    print(clusters_[i]) 
            av = [0, 1, 2, 3]
            waldo_ = [i for i, r_ in enumerate(records) if r_.split('_')[3]=='Waldo']
            l_waldo = list(clusters_[waldo_])
            i_w = max(set(l_waldo), key = l_waldo.count)
            av.remove(i_w)
            
            blank_ = [i for i, r_ in enumerate(records) if r_.split('_')[3]=='Blank']
            l_blank = list(clusters_[blank_])
            i_b = max(set(l_blank), key = l_blank.count)
            av.remove(i_b)
            
            puzzle_ = [i for i, r_ in enumerate(records) if r_.split('_')[3]=='Puzzle']
            l_puzzle = list(clusters_[puzzle_])
            i_p = max(set(l_puzzle), key = l_puzzle.count)
            av.remove(i_p)
            
            i_n = av[0]
             
            dict_lab = dict({i_w: 'Waldo', 
                             i_b: 'Blank',  
                             i_p: 'Puzzle',
                             i_n: 'Natural'})
            #dict_lab = dict({'Waldo': 0, 
            #                 'Blank': 1, 
            #                 'Puzzle': 3,
            #                 'Natural': 2})
            #print(dict_lab)
            ## Compute accuracy
            
            conf_mat = np.zeros((len(conditions), len(conditions))) 
            true_lab_c = 0
            for i, r_ in enumerate(records):
                true_lab = r_.split('_')[3]
                exp_lab = dict_lab[clusters_[i]]
           
                conf_mat[conditions_dict[true_lab], 
                         conditions_dict[exp_lab]] +=1
                
                if true_lab == exp_lab:
                    true_lab_c+=1
            accuracy = true_lab_c/len(records)
            accuracies.append(accuracy)
            print(subject)
            print('Accuracy: '+str(accuracy))
            print(conf_mat)
            
            disp=ConfusionMatrixDisplay(conf_mat, 
                                        display_labels=conditions, 
                                        )
            disp.plot(values_format='', 
                      colorbar=False, 
                      cmap='Blues')
       
            plt.show() 
            disp.figure_.savefig('output/ETRA/figures/clustering/longitudinal_knn_accuracy_{s_}.png'.format(s_=subject), 
                                 dpi=250)
            plt.clf()
            
            
        print('Mean accuracy: '+str(np.mean(accuracies)))
        
  
    def process_a_svm_dist(self):
        '''
        

        Returns
        -------
        None.

        '''
        
        conditions = sorted(self.config['data']['condition_set'])
        conditions_dict = dict({})
        for i, cond_ in enumerate(conditions):
            conditions_dict.update({cond_: i})
            
        binning = self.config['symbolization']['binning'] 
        symb = [f for f in self.symbolization_results if f.split('.')[0] == 'AoI'][0]
        with open(self.path+symb, 'rb') as f:
            symb = pickle.load(f)
        records = sorted(list(symb['recordings'].keys())) 
        records = [r_ for r_ in records if r_.split('_')[3] in conditions]
         
        y_ = []
        for record in records: 
            y_.append(record.split('_')[3])
         
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
             
            centers = symb['centers']
            centers_dict = dict({}) 
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
                
            for i in range(len(centers)):
                centers_dict.update({chr(i + 65): centers[i]})
              
            d_m, i_dict = aoi_dict_dist_mat(centers_dict,
                                            normalize=True) 
            ## Compute distance matrix for a given modality
            dist_mat = np.zeros((len(records), len(records))) 
            for j in range(1, len(records)): 
                for i in range(j):     
                    s_1 = record_dict[records[i]]
                    s_2 = record_dict[records[j]] 
                    ed = GeneralizedEditDistance(s_1, s_2, d_m, i_dict, self.config)
                    ed.process()
                    d_ = ed.dist_
                    dist_mat[i,j]=dist_mat[j,i]=d_
         
            dist_dict.update({type_: dist_mat})
        ## Compute the 'final' disatnce matrix
        t_dist=np.zeros((len(records), len(records)))
        for k_ in dist_dict.keys(): 
            t_dist += dist_dict[k_]**2
        t_dist=np.sqrt(t_dist)
        ordered_t_dist, _, _ = compute_serial_matrix(t_dist, 'ward')
        plt.imshow(ordered_t_dist)
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
       
        #1546
        
        for state in range(0, 200): 
            accuracies = []
            kf = KFold(n_splits=5, random_state=state, shuffle=True)
            conf_mat = np.zeros((len(conditions), len(conditions)))
            
            for i, (train_index, test_index) in enumerate(kf.split(records)): 
                X_train_r = records[train_index]
                X_test_r = records[test_index]
                
                y_train = y_[train_index]
                y_test = y_[test_index]
                
                X_train = np.array([X_embed[record_dict_idx[x_train_r]] for x_train_r in X_train_r])
                X_test = np.array([X_embed[record_dict_idx[x_test_r]] for x_test_r in X_test_r])
                 
                #clf = OneVsRestClassifier(SVC(C=1, kernel='rbf'))
                clf = SVC(C=1, kernel='rbf')
                clf.fit(X_train, y_train) 
                y_pred = clf.predict(X_test)
                
                correct=0
                for i in range(len(y_pred)): 
                    true_lab = y_test[i]
                    exp_lab = y_pred[i]
                    
                    conf_mat[conditions_dict[true_lab], 
                             conditions_dict[exp_lab]] +=1
                    
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
                                    display_labels=clf.classes_, 
                                    )
        disp.plot(values_format='', 
                  colorbar=False, 
                  cmap='Blues')
   
        plt.show() 
        disp.figure_.savefig('output/ETRA/figures/clustering/svm_accuracy_{n}.png'.format(n=len(conditions)), 
                             dpi=250)
        plt.clf()
    
 
        
    def process_a_knn(self):
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
        K = self.config['clustering']['k_nn']
        conditions = sorted(self.config['data']['condition_set'])
        conditions_dict = dict({})
        for i, cond_ in enumerate(conditions):
            conditions_dict.update({cond_: i})
            
        binning = self.config['symbolization']['binning'] 
        symb = [f for f in self.symbolization_results if f.split('.')[0] == 'AoI'][0]
        with open(self.path+symb, 'rb') as f:
            symb = pickle.load(f)
        records = sorted(list(symb['recordings'].keys())) 
        records = [r_ for r_ in records if r_.split('_')[3] in conditions]
         
        y_ = []
        for record in records: 
            y_.append(record.split('_')[3])
         
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
             
            centers = symb['centers']
            centers_dict = dict({}) 
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
                
            for i in range(len(centers)):
                centers_dict.update({chr(i + 65): centers[i]})
              
            d_m, i_dict = aoi_dict_dist_mat(centers_dict,
                                            normalize=True) 
            ## Compute distance matrix for a given modality
            dist_mat = np.zeros((len(records), len(records))) 
            for j in range(1, len(records)): 
                for i in range(j):     
                    s_1 = record_dict[records[i]]
                    s_2 = record_dict[records[j]] 
                    ed = GeneralizedEditDistance(s_1, s_2, d_m, i_dict, self.config)
                    ed.process()
                    d_ = ed.dist_
                    dist_mat[i,j]=dist_mat[j,i]=d_
         
            dist_dict.update({type_: dist_mat})
        ## Compute the 'final' disatnce matrix
        t_dist=np.zeros((len(records), len(records)))
        for k_ in dist_dict.keys(): 
            t_dist += dist_dict[k_]**2
        t_dist=np.sqrt(t_dist)
     
        best, best_s, best_confmat = 0, None, None 
        for state in range(0, 2000):
            accuracies = []
            kf = KFold(n_splits=5, random_state=state, shuffle=True)
            conf_mat = np.zeros((len(conditions), len(conditions)))
            
            for (train_index, test_index) in kf.split(records): 
                X_train = records[train_index]
                X_test = records[test_index]
                
                y_train = y_[train_index]
                y_test = y_[test_index]
                
                X_test_d = t_dist[test_index]
                X_test_d = X_test_d[:, train_index]
               
                correct=0
                for i, record_test in enumerate(X_test): 
                    idx = np.argpartition(X_test_d[i], K)[:K]
               
                    l_labels = list(y_train[idx]) 
                    exp_lab = max(set(l_labels), key = l_labels.count)
                    true_lab = y_test[i]
                    conf_mat[conditions_dict[true_lab], 
                             conditions_dict[exp_lab]] +=1
                    
                    if true_lab==exp_lab:
                        correct+=1 
                accuracy=correct/len(X_test)
                #print('Accuracy: ' + str(accuracy))  
                accuracies.append(accuracy)
            print('Mean accuracy: ' + str(np.mean(accuracies)) + ', for state ' + str(state))  
            if np.mean(accuracies) > best:
                best = np.mean(accuracies)
                best_s = state
                best_confmat = conf_mat
                
        print('Final accuracy: ' + str(best) + ', for state ' + str(best_s))
        print('Confusion matrix:')
        #print(best_confmat)  
      
        disp=ConfusionMatrixDisplay(best_confmat, 
                                    display_labels=conditions, 
                                    )
        disp.plot(values_format='', 
                  colorbar=False, 
                  cmap='Blues')
   
        plt.show()
        #fig = plt.gcf()
        disp.figure_.savefig('output/ETRA/figures/clustering/knn_accuracy_{n}.png'.format(n=len(conditions)), 
                             dpi=250)
        plt.clf()

 

def process(config, path, 
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
    if config['symbolization']['normalization'] == 'longitudinal':
        process_l(config, path, symbolization_results)
    elif config['symbolization']['normalization'] == 'all':
        process_a_svm_dist(config, path, symbolization_results)


def process_a_svm_dist(config, path, 
                       symbolization_results):
    
    conditions = sorted(config['data']['condition_set'])
    conditions_dict = dict({})
    for i, cond_ in enumerate(conditions):
        conditions_dict.update({cond_: i})
        
    binning = config['symbolization']['binning'] 
    symb = [f for f in symbolization_results if f.split('.')[0] == 'AoI'][0]
    with open(path+symb, 'rb') as f:
        symb = pickle.load(f)
    records = sorted(list(symb['recordings'].keys()))
 
    y_ = []
    for record in records: 
        y_.append(record.split('_')[3])
     
    records = np.array(records)
    y_ = np.array(y_)
    
    dist_dict = dict({})
    for type_ in ['oculomotorFixation',
                  'oculomotorSaccade', 
                  'scanpath', 
                  'AoI']:
        print('Processing {type_}...'.format(type_=type_))
        
        symb = [f for f in symbolization_results if f.split('.')[0] == type_][0]
        with open(path+symb, 'rb') as f:
            symb = pickle.load(f)
         
        centers = symb['centers']
        centers_dict = dict({}) 
        record_dict = dict({})
        record_dict_idx = dict({})
        
        for i, record in enumerate(records): 
            seq = symb['recordings'][record]['sequence']
            l_ = symb['recordings'][record]['lengths'] 
            seq_=[]
           
            if binning:
                for i in range(len(seq)):  
                    [seq_.append(chr(seq[i] + 65)) for _ in range((l_[i]))]
            else: 
                [seq_.append(chr(seq[i] + 65)) for i in range(len(seq))]
             
            
            record_dict.update({record: seq_})
            record_dict_idx.update({record: i})
            
        for i in range(len(centers)):
            centers_dict.update({chr(i + 65): centers[i]})
          
        d_m, i_dict = aoi_dict_dist_mat(centers_dict,
                                        normalize=True) 
        ## Compute distance matrix for a given modality
        dist_mat = np.zeros((len(records), len(records))) 
       
        for j in range(1, len(records)): 
            for i in range(j):     
                s_1 = record_dict[records[i]]
                s_2 = record_dict[records[j]] 
                ed = GeneralizedEditDistance(s_1, s_2, d_m, i_dict, config)
                ed.process()
                d_ = ed.dist_
                dist_mat[i,j]=dist_mat[j,i]=d_
     
        dist_dict.update({type_: dist_mat})
    ## Compute the 'final' disatnce matrix
    t_dist=np.zeros((len(records), len(records)))
    for k_ in dist_dict.keys(): 
        t_dist += dist_dict[k_]**2
    t_dist=np.sqrt(t_dist)
    
    ## Embedding in an enclidean metric space from distance matrix
    embedding = MDS(n_components=3, 
                    dissimilarity='precomputed', 
                    normalized_stress='auto')
    X_transformed = embedding.fit_transform(t_dist)
   
    h_dist = np.zeros_like(t_dist)
    l = h_dist.shape[0]
    
    v_sum = np.sum(t_dist**2, axis=0)/l
    h_sum = np.sum(t_dist**2, axis=1)/l
    _sum = np.sum(t_dist**2)/(l**2)
    
    for i in range(l):
        for j in range(l):
            h_dist[i,j] = t_dist[i,j]**2 - v_sum[j] - h_sum[i] + _sum
  
    h_dist *= -.5
    return 
    best, best_s, best_confmat = 0, None, None
    
    ## 1501 => .894
    for state in range(0, 250): 
        accuracies = []
        kf = KFold(n_splits=5, random_state=state, shuffle=True)
        conf_mat = np.zeros((len(conditions), len(conditions)))
        
        for i, (train_index, test_index) in enumerate(kf.split(records)): 
            X_train_r = records[train_index]
            X_test_r = records[test_index]
            
            y_train = y_[train_index]
            y_test = y_[test_index]
            
            #X_train = np.array([t_dist[record_dict_idx[x_train_r]] for x_train_r in X_train_r])
            #X_test = np.array([t_dist[record_dict_idx[x_test_r]] for x_test_r in X_test_r])
            
            X_train = np.array([h_dist[record_dict_idx[x_train_r]] for x_train_r in X_train_r])
            X_test = np.array([h_dist[record_dict_idx[x_test_r]] for x_test_r in X_test_r])
             
            clf = SVC(C=1, kernel='poly', degree=4)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
           
            correct=0
            for i in range(len(y_pred)): 
                true_lab = y_test[i]
                exp_lab = y_pred[i]
                
                conf_mat[conditions_dict[true_lab], 
                         conditions_dict[exp_lab]] +=1
                
                if true_lab==exp_lab:
                    correct+=1
                    
            accuracy=correct/len(y_pred)
            #print('Accuracy: ' + str(accuracy)) 
            accuracies.append(accuracy)
        #print('Final accuracy: ' + str(np.mean(accuracies)) + ', for state ' + str(state)) 
 
        if np.mean(accuracies) > best:
            best = np.mean(accuracies)
            best_s = state
            best_confmat = conf_mat
    print('Final accuracy: ' + str(best) + ', for state ' + str(best_s))
    print('Confusion matrix:')
    print(best_confmat)


def process_a_svm_centers(config, path, 
                          symbolization_results):
    
    binning = config['symbolization']['binning'] 
    symb = [f for f in symbolization_results if f.split('.')[0] == 'AoI'][0]
    with open(path+symb, 'rb') as f:
        symb = pickle.load(f)
    records = sorted(list(symb['recordings'].keys()))
 
    y_ = []
    for record in records: 
        y_.append(record.split('_')[3])
        
    records = np.array(records)
    y_ = np.array(y_)
 
    conditions = sorted(config['data']['condition_set'])
    conditions_dict = dict({})
    for i, cond_ in enumerate(conditions):
        conditions_dict.update({cond_: i})
      
    feature_dict = dict({})
    for i, record in enumerate(records): 
        features = []
        for type_ in ['oculomotorFixation',
                      'oculomotorSaccade', 
                      'scanpath', 
                      'AoI'
                      ]:
           
            symb = [f for f in symbolization_results if f.split('.')[0] == type_][0]
            with open(path+symb, 'rb') as f:
                symb = pickle.load(f)
             
            centers = symb['centers']
           
            seq = symb['recordings'][record]['sequence']
            l_features = np.array([centers[s_] for s_ in seq]) 
       
            features += list(l_features.flatten()) 
        feature_dict.update({record: np.array(features)})
   
    best, best_s, best_confmat = 0, None, None
   
    for state in range(651, 652):
        accuracies = []
        kf = KFold(n_splits=5, random_state=state, shuffle=True)
        conf_mat = np.zeros((len(conditions), len(conditions)))
        
        for i, (train_index, test_index) in enumerate(kf.split(records)): 
            X_train_r = records[train_index]
            X_test_r = records[test_index]
            
            y_train = y_[train_index]
            y_test = y_[test_index]
            
            X_train = np.array([feature_dict[x_train_r] for x_train_r in X_train_r])
            X_test = np.array([feature_dict[x_test_r] for x_test_r in X_test_r])
            
            clf = SVC(C=1, kernel='poly', degree=4, gamma=2)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
           
            correct=0
            for i in range(len(y_pred)): 
                true_lab = y_test[i]
                exp_lab = y_pred[i]
                
                conf_mat[conditions_dict[true_lab], 
                         conditions_dict[exp_lab]] +=1
                
                if true_lab==exp_lab:
                    correct+=1
                    
            accuracy=correct/len(y_pred)
            #print('Accuracy: ' + str(accuracy)) 
            accuracies.append(accuracy)
        #print('Final accuracy: ' + str(np.mean(accuracies)) + ', for state ' + str(state)) 
       
        if np.mean(accuracies) > best:
            best = np.mean(accuracies)
            best_s = state
            best_confmat = conf_mat
    print('Final accuracy: ' + str(best) + ', for state ' + str(best_s))
    print('Confusion matrix:')
    print(best_confmat)
   
        
def process_a(config, path, 
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
    K = config['clustering']['k_nn']['k_']
    binning = config['symbolization']['binning'] 
    ## Get all recordings
    symb = [f for f in symbolization_results if f.split('.')[0] == 'AoI'][0]
    with open(path+symb, 'rb') as f:
        symb = pickle.load(f)
    records = sorted(list(symb['recordings'].keys()))
 
    y_ = []
    for record in records: 
        y_.append(record.split('_')[3])
        
    records = np.array(records)
    y_ = np.array(y_)
 
      
    ## 26/49 => 0.871 for K=5 et k_=3
    ## 12 => 0.871 for K=5 et k_=3
    for state in range(100, 200):
        accuracies = []
        kf = KFold(n_splits=6, random_state=state, shuffle=True)
        
        for i, (train_index, test_index) in enumerate(kf.split(records)): 
            X_train = records[train_index]
            X_test = records[test_index]
            
            y_train = y_[train_index]
            y_test = y_[test_index]
            
            dist_dict = dict({})
            for record_test in X_test:
                dist_dict.update({record_test: dict({})})
                
            for type_ in ['oculomotorFixation',
                          'oculomotorSaccade', 
                          'scanpath', 
                          'AoI'
                          ]:
                #print('Processing {type_}...'.format(type_=type_))
                symb = [f for f in symbolization_results if f.split('.')[0] == type_][0]
                with open(path+symb, 'rb') as f:
                    symb = pickle.load(f)
                 
                centers = symb['centers']
                centers_dict = dict({}) 
                record_dict = dict({})
                
                for record in records: 
                    seq = symb['recordings'][record]['sequence']
                    l_ = symb['recordings'][record]['lengths'] 
                    seq_=[]
                   
                    if binning:
                        for i in range(len(seq)):  
                            [seq_.append(chr(seq[i] + 65)) for _ in range((l_[i]))]
                    else: 
                        [seq_.append(chr(seq[i] + 65)) for i in range(len(seq))]
                            
                    record_dict.update({record: seq_})
                    
                for i in range(len(centers)):
                    centers_dict.update({chr(i + 65): centers[i]})
                  
                d_m, i_dict = aoi_dict_dist_mat(centers_dict,
                                                normalize=True) 
                for record_test in X_test:
                    dist_array = np.zeros(len(X_train))
                    s_1 = record_dict[record_test]
                    for i, record_train in enumerate(X_train):
                        s_2 = record_dict[record_train]
                        ed = GeneralizedEditDistance(s_1, s_2, d_m, i_dict, config)
                        ed.process()
                        d_ = ed.dist_
                        dist_array[i]=d_
                        
                    dist_dict[record_test].update({type_: dist_array})
            
            correct=0
            for i, record_test in enumerate(X_test):
                t_dist = np.zeros(len(X_train))
                l_dist_dict = dist_dict[record_test]
                for k_ in l_dist_dict.keys():
                    t_dist += l_dist_dict[k_]**1
                idx = np.argpartition(t_dist, K)[:K]
           
                l_labels = list(np.array(y_train)[idx]) 
                label = max(set(l_labels), key = l_labels.count)
                
                if label==y_test[i]:
                    correct+=1 
                #else:
                #    print(record_test)
                #    print(label)
                # 
            accuracy=correct/len(X_test)
            #print('Accuracy: ' + str(accuracy))  
            accuracies.append(accuracy)
        print('Final accuracy: ' + str(np.mean(accuracies)) + ', for state ' + str(state))  
    return 


    for _ in range(1):
        X_train, X_test, y_train, y_test = train_test_split(records, y_, 
                                                            test_size=0.3,
                                                            random_state=4)
        #1, 4, 6, 7, 10, 13, 15
        #5, 8, 11
        dist_dict = dict({})
        for record_test in X_test:
            dist_dict.update({record_test: dict({})})
            
        for type_ in ['oculomotorFixation',
                      'oculomotorSaccade', 
                      'scanpath', 
                      'AoI']:
            #print('Processing {type_}...'.format(type_=type_))
            symb = [f for f in symbolization_results if f.split('.')[0] == type_][0]
            with open(path+symb, 'rb') as f:
                symb = pickle.load(f)
             
            centers = symb['centers']
            centers_dict = dict({}) 
            record_dict = dict({})
            
            for record in records: 
                seq = symb['recordings'][record]['sequence']
                l_ = symb['recordings'][record]['lengths'] 
                seq_=[]
               
                for i in range(len(seq)):  
                    [seq_.append(chr(seq[i] + 65)) for _ in range((l_[i]))]
          
                record_dict.update({record: seq_})
                
            for i in range(len(centers)):
                centers_dict.update({chr(i + 65): centers[i]})
                 
            for record_test in X_test:
                dist_array = np.zeros(len(X_train))
                s_1 = record_dict[record_test]
                for i, record_train in enumerate(X_train):
                    s_2 = record_dict[record_train]
                    ed = GeneralizedEditDistance(s_1, s_2, centers_dict, config)
                    ed.process()
                    d_ = ed.dist_
                    dist_array[i]=d_
                    
                dist_dict[record_test].update({type_: dist_array})
        
        correct=0
        for i, record_test in enumerate(X_test):
            t_dist = np.zeros(len(X_train))
            l_dist_dict = dist_dict[record_test]
            for k_ in l_dist_dict.keys():
                t_dist += l_dist_dict[k_]**1
            idx = np.argpartition(t_dist, K)[:K]
       
            l_labels = list(np.array(y_train)[idx]) 
            label = max(set(l_labels), key = l_labels.count)
            
            if label==y_test[i]:
                correct+=1 
            else:
                print(record_test)
                print(y_test[i])
             
        accuracy=correct/len(X_test)
        print('Accuracy: ' + str(accuracy))  
        
     
        
def process_l(config, path, 
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
    accuracies=[]
    for subject in config['data']['subject_set']: 
        to_process = [f for f in symbolization_results if f[:3] == subject]
      
        ## Load symbolization for each modality
        dist_dict=dict({})
        for type_ in ['oculomotorFixation',
                      'oculomotorSaccade', 
                      'scanpath', 
                      'AoI']:
            #print('Processing {type_}...'.format(type_=type_))
            symb = [f for f in to_process if f.split('.')[0].split('_')[1] == type_][0]
            with open(path+symb, 'rb') as f:
                symb = pickle.load(f)
             
            centers = symb['centers']
            centers_dict = dict({})
            records = sorted(list(symb['recordings'].keys()))
            record_dict = dict({})
            
            for record in records: 
                seq = symb['recordings'][record]['sequence']
                l_ = symb['recordings'][record]['lengths'] 
                seq_=[]
               
                for i in range(len(seq)):  
                    [seq_.append(chr(seq[i] + 65)) for _ in range((l_[i]))]
          
                record_dict.update({record: seq_})
                
            for i in range(len(centers)):
                centers_dict.update({chr(i + 65): centers[i]})
                 
            dist_mat = np.zeros((len(records), len(records)))
            
         
            for j in range(1, len(records)): 
                for i in range(j):     
                    s_1 = record_dict[records[i]]
                    s_2 = record_dict[records[j]] 
                    ed = GeneralizedEditDistance(s_1, s_2, centers_dict, config)
                    ed.process()
                    d_ = ed.dist_
                    dist_mat[i,j]=dist_mat[j,i]=d_
            #print(dist_mat[0])
            #print(records)
            #for i in range(len(dist_mat[0])):
            #    print(records[i])
            #    print(dist_mat[5,i])
            dist_dict.update({type_: dist_mat})
        
        t_dist=np.zeros((len(records), len(records)))
        for k_ in dist_dict.keys(): 
            t_dist += dist_dict[k_]
        #print('Clustering...')   
        km = kmedoids.KMedoids(4, method='fasterpam', 
                               random_state = 0)
        c = km.fit(t_dist)
         
        clusters_ = c.labels_ 
        center_idx_ = c.medoid_indices_
      
        
        #for i, r_ in enumerate(records):
        #    print(r_)
        #    print(clusters_[i])
        
     
        av = [0, 1, 2, 3]
        waldo_ = [i for i, r_ in enumerate(records) if r_.split('_')[3]=='Waldo']
        l_waldo = list(clusters_[waldo_])
        i_w = max(set(l_waldo), key = l_waldo.count)
        av.remove(i_w)
        
        blank_ = [i for i, r_ in enumerate(records) if r_.split('_')[3]=='Blank']
        l_blank = list(clusters_[blank_])
        i_b = max(set(l_blank), key = l_blank.count)
        av.remove(i_b)
        
        puzzle_ = [i for i, r_ in enumerate(records) if r_.split('_')[3]=='Puzzle']
        l_puzzle = list(clusters_[puzzle_])
        i_p = max(set(l_puzzle), key = l_puzzle.count)
        av.remove(i_p)
        
        i_n = av[0]
        
        
        
        dict_lab = dict({'Waldo': i_w, 
                         'Blank': i_b,  
                         'Puzzle': i_p,
                         'Natural': i_n})
        #dict_lab = dict({'Waldo': 0, 
        #                 'Blank': 1, 
        #                 'Puzzle': 3,
        #                 'Natural': 2})
        print(dict_lab)
        ## Compute accuracy
        true_lab_c = 0
        for i, r_ in enumerate(records):
            t_lab = dict_lab[r_.split('_')[3]]
            e_lab = clusters_[i]
  
            if t_lab == e_lab:
                true_lab_c+=1
        accuracy = true_lab_c/len(records)
        accuracies.append(accuracy)
        print('Accuracy: '+str(accuracy))
    print('\nMean accuracy: '+str(np.mean(accuracies)))
        
        
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
            
                 
    
            
            
            
            
            
            
            
            
            