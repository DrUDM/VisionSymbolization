# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd  
import copy 

import Vision as v
 
class BlockAnalysis():
    
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
        self.records = records 
        
        
    def process(self):
        '''
        

        Returns
        -------
        None.

        '''
        config = self.config
        for record in sorted(self.records):   
            subject, study, phase, level = record.split('.')[0].split('_')
            label = '_'.join([study, phase, level])
        
            if (subject in self.config['data']['subject_set'] and label in self.config['data']['label_set']):   
              #if subject=='0':
                print('\nAnalyzing file {rec_}'.format(rec_=record))   
                ## Load corresponding DataFrame
                df = pd.read_csv(self.path+record) 
                
                ## Compute segmentation for the full recording
                segmentation = v.BinarySegmentation(df, 
                                                    sampling_frequency = config['general']['sampling_frequency'], 
                                                    segmentation_method = config['general']['segmentation_method'],
                                                    distance_type = config['general']['distance_type'],
                                                    size_plan_x = config['general']['size_plan_x'],
                                                    size_plan_y = config['general']['size_plan_y'], 
                                                    status_threshold=.9,
                                                    IVT_velocity_threshold = 1,
                                                    verbose=False, 
                                                    display_segmentation=False)
              
                ## Compute featrures for each dimension
                if True:
                    self.process_oculomotor_features_(copy.deepcopy(segmentation), config, record)
                if True:
                    self.process_scanpath_features_(copy.deepcopy(segmentation), config, record)
                if True:
                    self.process_aoi_features_(copy.deepcopy(segmentation), config, record)
              
            
    def process_oculomotor_features_(self, 
                                     segmentation, config, record):
        '''
        
    
        Parameters
        ----------
        segmentation : TYPE
            DESCRIPTION.
        config : TYPE
            DESCRIPTION.
        record : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        '''  
        s_f = config['general']['sampling_frequency']
        interp = self.config['general']['interpolate_values']
        
        seg_shift = config['general']['oculomotor_seg_shift'] 
        seg_shift = int(np.ceil(seg_shift * s_f)) 
        
        part_length = config['general']['oculomotor_partition_length']
        part_length = int(np.ceil(part_length * s_f)) 
        
        n_s = segmentation.config['nb_samples'] - part_length
        nb_segments = int(n_s//seg_shift - 1)
        
        x_ = segmentation.data_set['x_array']
        y_ = segmentation.data_set['y_array']
       
        features = config['data']['oculomotor_features']
        result_df = pd.DataFrame(columns=features)
     
        for n_ in range(nb_segments): 
            try: 
                start=n_*seg_shift
                end=start+part_length
                
                l_segmentation = copy.deepcopy(segmentation)
                l_segmentation.new_segmentation_results(self.update_segmentation_results(segmentation.segmentation_results, 
                                                                                         start, end, 
                                                                                         x_, y_))
                l_segmentation.new_dataset(self.update_dataset(segmentation.data_set, 
                                                               start, end, 
                                                               config))
                l_segmentation.new_config(self.update_config(segmentation.config, 
                                                             start, end)) 
                fix_f = self.fixation_features(l_segmentation)
                sac_f = self.saccade_features(l_segmentation) 
                line = [start/s_f] + fix_f + sac_f 
                result_df.loc[len(result_df), :] = line
                
            except: 
                #print('Segment: {n_} rejected'.format(n_=n_))
                if interp:
                    line = [start/s_f] + [np.nan] * (len(features)-1) 
                    result_df.loc[len(result_df), :] = line
                pass 
            
     
        filename='output/results/ADABase/features/{r_}_oculomotor.csv'.format(r_=record.split('.')[0])
        result_df.to_csv(filename, index=False)
          
                
    def fixation_features(self, 
                          segmentation):
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


    def saccade_features(self, 
                         segmentation):
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
        sac_features.append(sac_a.saccade_efficiencies(get_raw=False)['efficiency_mean']) 
        sac_features.append(sac_a.saccade_peak_velocities(get_raw=False)['velocity_peak_mean'])  
        sac_features.append(sac_a.saccade_peak_accelerations(get_raw=False)['peak_acceleration_mean'])
        sac_features.append(sac_a.saccade_skewness_exponents(get_raw=False)['skewness_exponent_mean'])
        sac_features.append(sac_a.saccade_peak_velocity_amplitude_ratios(get_raw=False)['ratio_mean'])
          
        return sac_features

    
    def process_scanpath_features_(self,  
                                   segmentation, config, record):
        '''
        
    
        Parameters
        ----------
        segmentation : TYPE
            DESCRIPTION.
        config : TYPE
            DESCRIPTION.
        record : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        '''
        s_f = config['general']['sampling_frequency']
        interp = self.config['general']['interpolate_values']
        
        seg_shift = config['general']['scanpath_seg_shift'] 
        seg_shift = int(np.ceil(seg_shift * s_f)) 
        
        part_length = config['general']['scanpath_partition_length']
        part_length = int(np.ceil(part_length * s_f)) 
        
        n_s = segmentation.config['nb_samples'] - part_length
        nb_segments = int(n_s//seg_shift - 1)
        
        x_ = segmentation.data_set['x_array']
        y_ = segmentation.data_set['y_array']
        
        features = config['data']['scanpath_features']
        result_df = pd.DataFrame(columns=features)
     
        for n_ in range(nb_segments): 
            try:
                start=n_*seg_shift
                end=start+part_length
                
                l_segmentation = copy.deepcopy(segmentation)
                l_segmentation.new_segmentation_results(self.update_segmentation_results(segmentation.segmentation_results, 
                                                                                         start, end, 
                                                                                         x_, y_))
                l_segmentation.new_dataset(self.update_dataset(segmentation.data_set, 
                                                               start, end, 
                                                               config))
                l_segmentation.new_config(self.update_config(segmentation.config, 
                                                             start, end)) 
                sp_f = self.scanpath_features(l_segmentation) 
                line = [start/s_f] + sp_f 
                result_df.loc[len(result_df), :] = line
                
            except: 
                #print('Segment: {n_} rejected'.format(n_=n_))
                if interp:
                    line = [start/s_f] + [np.nan] * (len(features)-1) 
                    result_df.loc[len(result_df), :] = line
                pass 
     
        filename='output/results/ADABase/features/{r_}_scanpath.csv'.format(r_=record.split('.')[0])
        result_df.to_csv(filename, index=False)   
        
        
    def scanpath_features(self, 
                          segmentation):
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
        
        ## Compute geometrical descriptors
        geo_a = v.GeometricalAnalysis(scanpath, 
                                      verbose=False) 
        scan_features.append(geo_a.scanpath_length()['length'])
        scan_features.append(geo_a.scanpath_BCEA(BCEA_probability=.68, 
                                                 display_results=False)['BCEA'])
        scan_features.append(geo_a.scanpath_convex_hull(display_results=False, 
                                                        get_raw=False)['hull_area'])
        scan_features.append(geo_a.scanpath_HFD(HFD_hilbert_iterations=4, 
                                                HFD_k_max=10, 
                                                display_results=False, 
                                                get_raw=False)['fractal_dimension']) 
        ## Compute RQA descriptors
        rqa_a = v.RQAAnalysis(scanpath,  
                              verbose=False, 
                              display_results=False)
        scan_features.append(rqa_a.scanapath_RQA_recurrence_rate(display_results=False)['RQA_recurrence_rate'])
        scan_features.append(rqa_a.scanapath_RQA_laminarity(display_results=False)['RQA_laminarity']) 
        scan_features.append(rqa_a.scanapath_RQA_determinism(display_results=False)['RQA_determinism'])
        
        return scan_features

    
    def process_aoi_features_(self, 
                              segmentation, config, record):
        '''
        
    
        Parameters
        ----------
        segmentation : TYPE
            DESCRIPTION.
        config : TYPE
            DESCRIPTION.
        record : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        '''
        s_f = config['general']['sampling_frequency']
        interp = self.config['general']['interpolate_values']
        
        seg_shift = config['general']['aoi_seg_shift'] 
        seg_shift = int(np.ceil(seg_shift * s_f)) 
        
        part_length = config['general']['aoi_partition_length']
        part_length = int(np.ceil(part_length * s_f)) 
        
        n_s = segmentation.config['nb_samples'] - part_length
        nb_segments = int(n_s//seg_shift - 1)
        
        x_ = segmentation.data_set['x_array']
        y_ = segmentation.data_set['y_array']
        
        features = config['data']['aoi_features']
        result_df = pd.DataFrame(columns=features)
     
        for n_ in range(nb_segments):   
            try: 
                start=n_*seg_shift
                end=start+part_length
         
                l_segmentation = copy.deepcopy(segmentation)
                l_segmentation.new_segmentation_results(self.update_segmentation_results(segmentation.segmentation_results, 
                                                                                         start, end, 
                                                                                         x_, y_))
                l_segmentation.new_dataset(self.update_dataset(segmentation.data_set, 
                                                               start, end, 
                                                               config))
                l_segmentation.new_config(self.update_config(segmentation.config, 
                                                             start, end))  
                aoi_f = self.aoi_features(l_segmentation) 
                line = [start/s_f] + aoi_f 
                result_df.loc[len(result_df), :] = line
                
            except: 
                #print('Segment: {n_} rejected'.format(n_=n_))
                if interp:
                    line = [start/s_f] + [np.nan] * (len(features)-1) 
                    result_df.loc[len(result_df), :] = line
                pass 
                 
        filename='output/results/ADABase/features/{r_}_AoI.csv'.format(r_=record.split('.')[0])
        result_df.to_csv(filename, index=False)  
        
        
    def aoi_features(self, 
                     segmentation):
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
        aoi_features.append(basic_a.AoI_BCEA(BCEA_probability=.68, 
                                             get_raw=False)['average_BCEA'])
        aoi_features.append(basic_a.AoI_BCEA(BCEA_probability=.68, 
                                             get_raw=False)['disp_BCEA'])  
        ## Add lempl ziv complexity 
        lz = v.LemplZiv(aoi)
        aoi_features.append(lz.results['AoI_lempel_ziv_complexity'])
     
        ## Compute various entropies
        markov_a = v.MarkovBasedAnalysis(aoi, 
                                         verbose=False, 
                                         display_results=False, 
                                         display_AoI_identification=False)  
        entropies = markov_a.AoI_transition_entropy() 
        aoi_features.append(np.exp(entropies['AoI_transition_stationary_entropy']))
        aoi_features.append(np.exp(entropies['AoI_transition_joint_entropy']))
        aoi_features.append(np.exp(entropies['AoI_transition_conditional_entropy']))
        aoi_features.append(np.exp(entropies['AoI_transition_mutual_information']))
         
        return aoi_features


    def update_config(self, 
                      config, start, end): 
        '''
        
    
        Parameters
        ----------
        config : TYPE
            DESCRIPTION.
        start : TYPE
            DESCRIPTION.
        end : TYPE
            DESCRIPTION.
    
        Returns
        -------
        config : TYPE
            DESCRIPTION.
    
        '''
        config['nb_samples'] = end-start
        return config
     
    
    def update_dataset(self, 
                       data_set, start, end, 
                       config): 
        '''
        
    
        Parameters
        ----------
        data_set : TYPE
            DESCRIPTION.
        start : TYPE
            DESCRIPTION.
        end : TYPE
            DESCRIPTION.
    
        Returns
        -------
        TYPE
            DESCRIPTION.
    
        '''
        n_dataset = dict({
            'x_array': data_set['x_array'][start:end] ,
            'y_array': data_set['y_array'][start:end], 
            'z_array': data_set['z_array'][start:end],
            'status': data_set['status'][start:end] , 
            'absolute_speed': data_set['absolute_speed'][start:end]
                })
        
        ## To avoid analysis of empty segments 
        prop = config['general']['available_value_prop']
        pr_ = np.sum(n_dataset['status'])/len(n_dataset['status']) 
        assert pr_ >= prop, 'Too many missing values for this segment'
        
        return n_dataset
      
        
    def update_segmentation_results(self, 
                                    segmentation_results, 
                                    start, end, 
                                    x_, y_):
        '''
        
    
        Parameters
        ----------
        segmentation_results : TYPE
            DESCRIPTION.
        start : TYPE
            DESCRIPTION.
        end : TYPE
            DESCRIPTION.
        x_ : TYPE
            DESCRIPTION.
        y_ : TYPE
            DESCRIPTION.
    
        Returns
        -------
        TYPE
            DESCRIPTION.
    
        '''
        from Vision.utils.segmentation_utils import centroids_from_ints
     
        f_ints = segmentation_results['fixation_intervals']
        s_ints = segmentation_results['saccade_intervals']
        n_f_ints = [f_int for f_int in f_ints if (f_int[0]>=start and f_int[1]<end)]
        n_s_ints = [s_int for s_int in s_ints if (s_int[0]>=start and s_int[1]<end)]
        n_ctrds = centroids_from_ints(n_f_ints,
                                      x_, y_) 
        n_f_ints = [list(np.array(n_f_int)-start) for n_f_int in n_f_ints]
        n_s_ints = [list(np.array(n_s_int)-start) for n_s_int in n_s_ints]
        i_lab = segmentation_results['is_labeled']
        n_i_lab = i_lab[start: end]
     
        return dict({
            'is_labeled': n_i_lab,
            'fixation_intervals': n_f_ints, 
            'saccade_intervals': n_s_ints,
            'centroids': n_ctrds, 
                })
           
                    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            