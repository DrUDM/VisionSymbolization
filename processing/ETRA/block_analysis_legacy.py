# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import math 

import Vision as v



def process(config, path, records):
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
    if False:
        process_oculomotor_features(config, path, records)
    if False:
        process_scanpath_features(config, path, records)
    if True:
        process_aoi_features(config, path, records)
    


def process_aoi_features(config, path, records):
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
    for record in records:  
        subject, trial, task, condition, stimulus = record.split('.')[0].split('_')
     
        if (subject in config['data']['subject_set']
            and task in config['data']['task_set']
            and condition in config['data']['condition_set']):
          
        
            print('\n\nAnalyzing file {rec_}'.format(rec_=record)) 
            meta_data = dict({'subject': subject,
                              'trial': trial,
                              'task': task,
                              'condition': condition,
                              'stimulus': stimulus}) 
            df = pd.read_csv(path+record)
            
            aoi_part_length = config['general']['aoi_partition_length']
            s_f = config['general']['sampling_frequency']
            
            aoi_part_idx_length = int(np.ceil(aoi_part_length * s_f)) 
            aoi_nb_split = int(len(df)/aoi_part_idx_length)
            
            features = config['data']['aoi_features']
            result_df = pd.DataFrame(columns=features)
             
            for i in range(aoi_nb_split):   
                #try:
                    print('\nAnalysing block: {i_}...'.format(i_=i))
                    p_df = df.iloc[i*aoi_part_idx_length: 
                                   (i+1)*aoi_part_idx_length]
                    
                    sp = v.Scanpath(p_df, 
                                    sampling_frequency = config['general']['sampling_frequency'], 
                                    segmentation_method = config['general']['segmentation_method'],
                                    distance_type = config['general']['distance_type'],
                                    size_plan_x = config['general']['size_plan_x'],
                                    size_plan_y = config['general']['size_plan_y'],
                                    display_results=False,  
                                    verbose=False)
                    comp_features = aoi_features(sp)
                    for i, feature in enumerate(comp_features): 
                        assert not math.isnan(feature)
                    line = [p_df.iloc[0]['timestamp']]
                    line += comp_features 
                    result_df.loc[len(result_df), :] = line
                
                    print('...done')
                    
                #except Exception: 
                #    print('...rejected') 
                #    line = [p_df.iloc[0]['timestamp']]
                #    line += [np.nan] * (len(features)-1)
                #   
                #    result_df.loc[len(result_df), :] = line
                #    pass
                
            filename = 'output/results/ETRA/features/{sub_}_{tri_}_{tas_}_{con_}_{sti_}_AoI.csv'.format(sub_=meta_data['subject'],
                                                                                                        tri_=meta_data['trial'],
                                                                                                        tas_=meta_data['task'],
                                                                                                        con_=meta_data['condition'],
                                                                                                        sti_=meta_data['stimulus'])
            result_df.to_csv(filename, index=False)
            
            
def process_scanpath_features(config, path, records):
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
    for record in records: 
        subject, trial, task, condition, stimulus = record.split('.')[0].split('_')
        
        if (subject in config['data']['subject_set']
            and task in config['data']['task_set']
            and condition in config['data']['condition_set']):
          
            print('\n\nAnalyzing file {rec_}'.format(rec_=record))
                 
            meta_data = dict({'subject': subject,
                              'trial': trial,
                              'task': task,
                              'condition': condition,
                              'stimulus': stimulus}) 
            df = pd.read_csv(path+record)
            
            sp_part_length = config['general']['scanpath_partition_length']
            s_f = config['general']['sampling_frequency']
            
            sp_part_idx_length = int(np.ceil(sp_part_length * s_f)) 
            sp_nb_split = int(len(df)/sp_part_idx_length)
            
            features = config['data']['scanpath_features']
            result_df = pd.DataFrame(columns=features)
         
            for i in range(sp_nb_split):  
                try:
                    print('\nAnalysing block: {i_}...'.format(i_=i))
                    p_df = df.iloc[i*sp_part_idx_length: 
                                   (i+1)*sp_part_idx_length]
                    
                    segmentation = v.BinarySegmentation(p_df, 
                                                        sampling_frequency = config['general']['sampling_frequency'], 
                                                        segmentation_method = config['general']['segmentation_method'],
                                                        distance_type = config['general']['distance_type'],
                                                        size_plan_x = config['general']['size_plan_x'],
                                                        size_plan_y = config['general']['size_plan_y'], 
                                                        verbose=False)
                    comp_features = scanpath_features(segmentation)
                    for i, feature in enumerate(comp_features): 
                        assert not math.isnan(feature)
                    line = [p_df.iloc[0]['timestamp']]
                    line += comp_features 
                    result_df.loc[len(result_df), :] = line
                    
                    print('...done')
                    
                except Exception: 
                    print('...rejected') 
                    line = [p_df.iloc[0]['timestamp']]
                    line += [np.nan] * (len(features)-1)
                   
                    result_df.loc[len(result_df), :] = line
                    pass
               
            filename = 'output/results/ETRA/features/{sub_}_{tri_}_{tas_}_{con_}_{sti_}_scanpath.csv'.format(sub_=meta_data['subject'],
                                                                                                      tri_=meta_data['trial'],
                                                                                                      tas_=meta_data['task'],
                                                                                                      con_=meta_data['condition'],
                                                                                                      sti_=meta_data['stimulus'])
            result_df.to_csv(filename, index=False)
            
    
def process_oculomotor_features(config, path, records):
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
    for record in records: 
        subject, trial, task, condition, stimulus = record.split('.')[0].split('_')
        
        if (subject in config['data']['subject_set']
            and task in config['data']['task_set']
            and condition in config['data']['condition_set']):
        
            print('\n\nAnalyzing file {rec_}'.format(rec_=record))
                 
            meta_data = dict({'subject': subject,
                              'trial': trial,
                              'task': task,
                              'condition': condition,
                              'stimulus': stimulus}) 
            df = pd.read_csv(path+record)
            
            oc_part_length = config['general']['oculomotor_partition_length']
            s_f = config['general']['sampling_frequency']
            
            oc_part_idx_length = int(np.ceil(oc_part_length * s_f)) 
            oc_nb_split = int(len(df)/oc_part_idx_length)
            
            features = config['data']['oculomotor_features']
            result_df = pd.DataFrame(columns=features)
         
            for i in range(oc_nb_split): 
                try: 
                    print('\nAnalysing block: {i_}...'.format(i_=i))
                    p_df = df.iloc[i*oc_part_idx_length: 
                                   (i+1)*oc_part_idx_length]
                
                    segmentation = v.BinarySegmentation(p_df, 
                                                        sampling_frequency = config['general']['sampling_frequency'], 
                                                        segmentation_method = config['general']['segmentation_method'],
                                                        distance_type = config['general']['distance_type'],
                                                        size_plan_x = config['general']['size_plan_x'],
                                                        size_plan_y = config['general']['size_plan_y'], 
                                                        verbose=False)
                 
                    fix_f = fixation_features(segmentation)
                    sac_f = saccade_features(segmentation)
                    comp_features = fix_f + sac_f   
                    for i, feature in enumerate(comp_features): 
                        assert not math.isnan(feature)
           
                    line = [p_df.iloc[0]['timestamp']]
                    line += comp_features 
                    result_df.loc[len(result_df), :] = line
                    
                    print('...done')
                    
                except Exception: 
                    print('...rejected') 
                    line = [p_df.iloc[0]['timestamp']]
                    line += [np.nan] * (len(features)-1)
                   
                    result_df.loc[len(result_df), :] = line
                    pass
                
            filename = 'output/results/ETRA/features/{sub_}_{tri_}_{tas_}_{con_}_{sti_}_oculomotor.csv'.format(sub_=meta_data['subject'],
                                                                                                      tri_=meta_data['trial'],
                                                                                                      tas_=meta_data['task'],
                                                                                                      con_=meta_data['condition'],
                                                                                                      sti_=meta_data['stimulus'])
            result_df.to_csv(filename, index=False)
  

def fixation_features(segmentation):
    '''
    

    Parameters
    ----------
    segmentation : TYPE
        DESCRIPTION.

    Returns
    -------
    fix_features : TYPE
        DESCRIPTION.

    ''' 
    fix_a = v.FixationAnalysis(segmentation, 
                               verbose=False) 
    fix_features = []
    fix_features.append(fix_a.fixation_frequency_wrt_labels()['frequency'])  
    fix_features.append(fix_a.fixation_average_velocity_means(weighted=False, 
                                                              get_raw=False)['average_velocity_means']) 
    fix_features.append(fix_a.fixation_drift_displacements(get_raw=False)['drift_displacement_mean']) 
    fix_features.append(fix_a.fixation_drift_distances(get_raw=False)['drift_cumul_distance_mean']) 
    fix_features.append(fix_a.fixation_drift_velocities(get_raw=False)['drift_velocity_mean']) 
    fix_features.append(fix_a.fixation_BCEA(BCEA_probability=.68, 
                                            get_raw=False)['average_BCEA']) 
    return fix_features

 
def saccade_features(segmentation):
    '''
    

    Parameters
    ----------
    segmentation : TYPE
        DESCRIPTION.

    Returns
    -------
    sac_features : TYPE
        DESCRIPTION.

    '''
 
    sac_a = v.SaccadeAnalysis(segmentation, 
                              verbose=False) 
    sac_features = []
    
    sac_features.append(sac_a.saccade_frequency_wrt_labels()['frequency'])   
    sac_features.append(sac_a.saccade_amplitudes(get_raw=False)['amplitude_mean'])
    sac_features.append(sac_a.saccade_travel_distances(get_raw=False)['distance_mean'])
    sac_features.append(sac_a.saccade_efficiencies(get_raw=False)['efficiency_mean'])
    sac_features.append(sac_a.saccade_horizontal_deviations(absolute=True, 
                                                            get_raw=False)['horizontal_deviation_mean'])
    sac_features.append(sac_a.saccade_successive_deviations(get_raw=False)['successive_deviation_mean'])
    sac_features.append(sac_a.saccade_max_curvatures(get_raw=False)['max_curvature_mean']) 
    sac_features.append(sac_a.saccade_peak_velocities(get_raw=False)['velocity_peak_mean'])  
    sac_features.append(sac_a.saccade_peak_accelerations(get_raw=False)['peak_acceleration_mean'])
    sac_features.append(sac_a.saccade_peak_velocity_amplitude_ratios(get_raw=False)['ratio_mean'])
      
    return sac_features


def scanpath_features(segmentation):
    '''
    

    Parameters
    ----------
    segmentation : TYPE
        DESCRIPTION.

    Returns
    -------
    scan_features : TYPE
        DESCRIPTION.

    '''
   
    scan_features = []
    scanpath = v.Scanpath(segmentation, 
                          display_scanpath=False, 
                          verbose=False)
    
    geo_a = v.GeometricalAnalysis(scanpath, 
                                  verbose=False) 
    scan_features.append(geo_a.scanpath_BCEA(BCEA_probability=.68, 
                                             display_results=False)['BCEA'])
    scan_features.append(geo_a.scanpath_convex_hull(display_results=False, 
                                                    get_raw=False)['hull_area'])
    scan_features.append(geo_a.scanpath_HFD(HFD_hilbert_iterations=4, 
                                            HFD_k_max=10, 
                                            display_results=False, 
                                            get_raw=False)['fractal_dimension'])
    scan_features.append(np.exp(geo_a.scanpath_k_coefficient(display_results=False)['k_coefficient']))
    scan_features.append(geo_a.scanpath_voronoi_cells(display_results=False, 
                                                      get_raw=False)['gamma_parameter'])
    
    #rqa_a = v.RQAAnalysis(scanpath)
    #scan_features.append(rqa_a.scanapath_RQA_recurrence_rate(display_results=False)['RQA_recurrence_rate'])
    #scan_features.append(rqa_a.scanapath_RQA_laminarity(display_results=False)['RQA_laminarity']) 
    #scan_features.append(rqa_a.scanapath_RQA_determinism(display_results=False)['RQA_determinism'])
    
    return scan_features
    
    
def aoi_features(segmentation):
    '''
    

    Parameters
    ----------
    sp : TYPE
        DESCRIPTION.

    Returns
    -------
    aoi_features : TYPE
        DESCRIPTION.

    ''' 
    aoi_features = []
    aoi = v.AoISequence(segmentation, 
                        AoI_identification_method='I_AP',
                        display_AoI_identification=False,  
                        verbose=False) 
    ## Add basic descriptors
    basic_a = v.AoIBasicAnalysis(aoi, 
                                 verbose=False)
    aoi_features.append(basic_a.AoI_count()['count'])
    aoi_features.append(basic_a.AoIBCEA(BCEA_probability=.68, 
                                        get_raw=False)['average_BCEA'])
    #aoi_features.append(basic_a.AoIBCEA(BCEA_probability=.68, 
    #                                    get_raw=False)['variance_BCEA'])
 
    ## Compute various entropies
    markov_a = v.MarkovBasedAnalysis(aoi, 
                                     verbose=False, 
                                     display_results=False, 
                                     display_AoI_identification=False)  
    entropies = markov_a.AoI_transition_entropy() 
    aoi_features.append(np.exp(entropies['AoI_transition_stationary_entropy']))
    aoi_features.append(np.exp(entropies['AoI_transition_joint_entropy']))
    aoi_features.append(np.exp(entropies['AoI_transition_conditional_entropy']))
    #aoi_features.append(np.exp(entropies['AoI_transition_mutual_information']))
     
    return aoi_features
    
    
    
 