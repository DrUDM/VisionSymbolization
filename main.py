# -*- coding: utf-8 -*-

import Vision as v
import numpy as np
 
root = 'dataset/'
np.random.seed(1)
 

## For Bineary Segmentation

#bs = v.BinarySegmentation(root + 'data_1.csv', 
#                         sampling_frequency = 256,  
#                         segmentation_method = 'I_HMM',
#                         distance_type = 'angular',                        
#                         display_segmentation = True,
#                         size_plan_x = 1200,
#                         size_plan_y = 800)

## For Saccades

#sa = v.SaccadeAnalysis(bs)

#print(v.saccade_main_sequence(sa))
#print(v.saccade_main_sequence(bs,
#                                      get_raw = True, 
#                                      saccade_weighted_average_acceleration_profiles = True))

 
#

#print(v.saccade_main_sequence(root + 'data_1.csv', 
#                                      sampling_frequency = 256, 
#                                      segmentation_method = 'I_HMM',
#                                      distance_type = 'angular',                        
#                                      distance_type = 'angular',                        
#                                      display_segmentation = True,
#                                      size_plan_x = 1200,
#                                      size_plan_y = 800,
#                                      get_raw = False))


## For Fixations
 
#fa = v.FixationAnalysis(root + 'data_1.csv', 
#                         sampling_frequency = 256,  
#                         segmentation_method = 'I_HMM',
#                         distance_type = 'angular',                        
#                         display_segmentation = True,
#                         size_plan_x = 1200,
#                         size_plan_y = 800)
#print('THEN THEN')
#print(v.fixation_BCEA(root + 'data_1.csv', 
#                         sampling_frequency = 256,  
#                         segmentation_method = 'I_HMM',
#                         distance_type = 'angular',                        
#                         display_segmentation = True,
#                         size_plan_x = 1200,
#                         size_plan_y = 800))
#print(v.fixation_BCEA(bs,
#                                get_raw = True,
#                                fixation_weighted_average_velocity_means = True, 
#                                fixation_BCEA_probability = 0.75))
#print(v.fixation_BCEA(root + 'data_1.csv', 
#                                      sampling_frequency = 256, 
#                                      segmentation_method = 'I_HMM',
#                                      distance_type = 'angular',   
#                                      display_segmentation = True,
#                                      size_plan_x = 1200,
#                                      size_plan_y = 800))
 

## For Signal-based

#sb = v.CrossFrequencyAnalysis([root + 'data_1.csv', root + 'data_2.csv'], 
#                                      sampling_frequency = 256, 
#                                      segmentation_method = 'I_HMM',
#                                      distance_type = 'angular',   
#                                      display_segmentation = True,
#                                      size_plan_x = 1200,
#                                      size_plan_y = 800)

#print(v.signal_coherency(sb, 
#                          Welch_samples_per_segment = 100))
#print(v.signal_coherency([root + 'data_1.csv', root + 'data_2.csv'], 
#                                      sampling_frequency = 256, 
#                                      segmentation_method = 'I_HMM',
#                                      distance_type = 'angular',   
#                                      display_segmentation = True,
#                                      size_plan_x = 1200,
#                                      size_plan_y = 800))

#sb = v.StochasticAnalysis(root + 'data_1.csv', 
#                                      sampling_frequency = 256, 
#                                      segmentation_method = 'I_HMM',
#                                      distance_type = 'angular',   
#                                      display_segmentation = True,
#                                      size_plan_x = 1200,
#                                      size_plan_y = 800)
#print(v.DACF(sb))
#print(v.DACF(root + 'data_1.csv', 
#                                      sampling_frequency = 256, 
#                                      segmentation_method = 'I_HMM',
#                                      distance_type = 'angular',   
#                                      display_segmentation = True,
#                                      size_plan_x = 1200,
#                                      size_plan_y = 800))


## For Scanpath 
 

#sp = v.Scanpath(bs, 
#                display_scanpath=True)

#sp = v.Scanpath(root + 'data_1.csv', 
#                sampling_frequency = 256,  
#                segmentation_method = 'I_HMM',
#                distance_type = 'angular',                        
#                display_segmentation = True,
#                size_plan_x = 1200,
#                size_plan_y = 800,
#                display_scanpath=True)

 
## For Geometrical Analysis  
 
#ga = v.GeometricalAnalysis(root + 'data_1.csv', 
#                           sampling_frequency = 256, 
#                           segmentation_method = 'I_HMM',
#                           distance_type = 'angular',                        
#                           display_segmentation = True,
#                           display_scanpath=True,
#                           size_plan_x = 1200,
#                           size_plan_y = 800)
 
#ga = v.GeometricalAnalysis(bs,
#                           display_scanpath=True)
 
#ga = v.GeometricalAnalysis(sp)
#
 
#print(v.scanpath_HFD(bs))
#                     display_results=False))
#print(v.scanpath_HFD(root + 'data_1.csv', 
#                           sampling_frequency = 256, 
#                           segmentation_method = 'I_HMM',
#                           distance_type = 'angular',                        
#                           display_segmentation = True,
#                           display_scanpath=True,
#                           size_plan_x = 1200,
#                           size_plan_y = 800))


## For RQA Analysis  

#ga = v.RQAAnalysis(root + 'data_1.csv', 
#                           sampling_frequency = 256, 
#                           segmentation_method = 'I_HMM',
#                           distance_type = 'angular',                        
#                           display_segmentation = True,
#                           display_scanpath=True,
#                           size_plan_x = 1200,
#                           size_plan_y = 800)
 
#ga = v.RQAAnalysis(bs,
#                   display_scanpath=True)

#ga = v.RQAAnalysis(sp)

 
#print(v.scanapath_RQA_entropy(bs))
#print('THEN')
#print(v.scanapath_RQA_entropy(sp))

#print(v.scanapath_RQA_recurrence_rate(root + 'data_1.csv', 
#                           sampling_frequency = 256, 
#                           segmentation_method = 'I_HMM',
#                           distance_type = 'angular',                        
#                           display_segmentation = True,
#                           display_scanpath=True,
#                           size_plan_x = 1200,
#                           size_plan_y = 800))


## For Point Mapping Distance

sp1 = v.Scanpath(root + 'data_1.csv', 
                sampling_frequency = 256,                  
                segmentation_method = 'I_HMM',
                distance_type = 'angular',                        
                display_segmentation = False,
                size_plan_x = 1200,
                size_plan_y = 800,
                display_scanpath=False,
                verbose=False)

sp2 = v.Scanpath(root + 'data_2.csv', 
                sampling_frequency = 256,  
                segmentation_method = 'I_HMM',
                distance_type = 'angular',                        
                display_segmentation = False,
                size_plan_x = 1200,
                size_plan_y = 800,
                display_scanpath=True,
                verbose=True)

#pm = v.PointMappingDistance([sp1, sp2])

#print(v.frechet_distance([sp1, sp2]))

## For Elastic Distance

#sp1 = v.Scanpath(root + 'data_1.csv', 
#                sampling_frequency = 256,                  
#                segmentation_method = 'I_HMM',
#                distance_type = 'angular',                        
#                display_segmentation = False,
#                size_plan_x = 1200,
#                size_plan_y = 800,
#                display_scanpath=True,
#                verbose=False)

#sp2 = v.Scanpath(root + 'data_2.csv', 
#                sampling_frequency = 256,  
#                segmentation_method = 'I_HMM',
#                distance_type = 'angular',                        
#                display_segmentation = False,
#                size_plan_x = 1200,
#                size_plan_y = 800,
#                display_scanpath=True,
#                verbose=True)

#pm = v.ElasticDistance([sp1, sp2])

#print(v.DTW_distance([sp1, sp2]))
                                         
## For Edit Distance
    
#sp1 = v.Scanpath(root + 'data_1.csv', 
#                sampling_frequency = 256,                  
#                segmentation_method = 'I_HMM',
#                distance_type = 'angular',                        
#                display_segmentation = False,
#                size_plan_x = 1200,
#                size_plan_y = 800,
#                size_plan_y = 800,
#                size_plan_y = 800,
#                display_scanpath=True,
#                verbose=False)

#sp2 = v.Scanpath(root + 'data_2.csv', 
#                sampling_frequency = 256,  
#                segmentation_method = 'I_HMM',
#                distance_type = 'angular',                        
#                display_segmentation = False,
#                size_plan_x = 1200,
#                size_plan_y = 800,
#                display_scanpath=True,
#                verbose=True)

sp3 = v.Scanpath(root + 'data_3.csv', 
                sampling_frequency = 256,                  
                segmentation_method = 'I_HMM',
                distance_type = 'angular',                        
                display_segmentation = False,
                size_plan_x = 1200,
                size_plan_y = 800,
                display_scanpath=True,
                verbose=False)

sp4 = v.Scanpath(root + 'data_4.csv', 
                sampling_frequency = 256,                  
                segmentation_method = 'I_HMM',
                distance_type = 'angular',                        
                display_segmentation = False,
                size_plan_x = 1200,
                size_plan_y = 800,
                display_scanpath=True,
                verbose=False)

sp5 = v.Scanpath(root + 'data_5.csv', 
                sampling_frequency = 256,                  
                segmentation_method = 'I_HMM',
                distance_type = 'angular',                        
                display_segmentation = False,
                size_plan_x = 1200,
                size_plan_y = 800,
                display_scanpath=True,
                verbose=False)

sp6 = v.Scanpath(root + 'data_6.csv', 
                sampling_frequency = 256,                  
                segmentation_method = 'I_HMM',
                distance_type = 'angular',                        
                display_segmentation = False,
                size_plan_x = 1200,
                size_plan_y = 800,
                display_scanpath=True,
                verbose=False)

#pm = v.StringEditDistance(  [sp1, sp2])

#print(v.scanpath_generalized_edit_distance([sp1, sp2], 
#                                  scanpath_temporal_binning=False, 
#                                  display_results = False))

## For scanmatch
#print(v.scanmatch_score([sp1, sp2]))

## For subsmatch
#print(v.subsmatch_similarity([sp1, sp2]))

## For multimatch
#print(v.multimatch_alignment([sp1, sp2], 
#                             display_results = False)
#      )


## For AoI Sequence
#aoi_s = v.AoISequence(sp1, 
#                      AoI_identification_method='I_KM', 
#                      AoI_IKM_cluster_number='search', 
#                      )

#aoi_s = v.AoISequence(sp1, 
#                       AoI_identification_method='I_HMM')
#print(aoi_s.centers)
#mb_a = v.MarkovBasedAnalysis(sp1)

#print(v.AoI_HMM(sp1))
#model = v.AoI_HMM(sp1)['AoI_HMM_model_instance']


#val = sp1.values[:2]
#val += np.random.random(val.shape)*10

#print(v.AoI_HMM_fisher_vector(np.tile(val, 10), 
#      AoI_HMM_model = model))


#print(np.tile(val, 2).shape)
#print(v.AoI_transition_matrix(mb_a))
 

print('HERE')
l_ = v.scanpath_length(sp1)
print(l_)

               
#seqs = [['A', 'B', 'B', 'A'], 
#        ['A', 'B', 'B', 'A', 'C', 'A'], 
#        ['A', 'B', 'D']]

#AoI_durations = [np.random.random(4), 
#                 np.random.random(6), 
#                 np.random.random(3)]
             
#seqs = [sp1, sp2, sp3, sp4, sp5, sp6]               
#aoi_seqs = v.AoI_sequences(seqs, 
#                      display_scanpath=True, 
#                      AoI_identification_method = 'I_KM', 
#                      AoI_IKM_cluster_number = 5,
#                      AoI_temporal_binning = False, 
#                      AoI_temporal_binning_length=.2,  
#                      AoI_durations=AoI_durations) 
 

#print(aoi_seqs[0].fixation_analysis.segmentation_results)
#print(v.AoI_eMine(aoi_seqs))
#print(v.AoI_trend_analysis(aoi_seqs))
#print(v.AoI_constrained_DTW_barycenter_averaging(aoi_seqs))
 
#aoi1 = v.AoISequence(sp1, 
#                     AoI_identification_method = 'I_HMM')
#print(aoi1.durations)
#[aoi1, aoi2] = v.AoI_sequences([sp1, sp2]) 
#print(v.AoI_smith_waterman([sp1, sp2]))
#print(v.AoI_longest_common_subsequence([aoi1, aoi2]))

#print(v.AoI_eMine([sp1, sp2, sp3]))


## For visualizations 

#v.AoI_scarf_plot(aoi_seqs, 
#                 AoI_scarf_plot_save='scarf_plot')
#v.AoI_sankey_diagram(aoi_seqs  ,
#                     AoI_sankey_diagram_save='raw_sakey')

#v.AoI_time_plot(aoi_seqs[0])



