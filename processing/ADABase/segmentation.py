# -*- coding: utf-8 -*-

import numpy as np
import ruptures as rpt 
import pandas as pd

import matplotlib.pyplot as plt 


class Segmentation():
    
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
            print('Segmenting oculomotor features...')
            oculomotor_feature_records = [feature_record for feature_record in self.feature_records
                                          if feature_record.split('.')[0].split('_')[-1] == 'oculomotor']
            self.signal_segmentation(oculomotor_feature_records, 
                                     'oculomotor') 
        ## For scanpath features only
        if True:
            print('Segmenting scanpath features...')
            scanpath_feature_records = [feature_record for feature_record in self.feature_records
                                        if feature_record.split('.')[0].split('_')[-1] == 'scanpath']
            self.signal_segmentation(scanpath_feature_records, 
                                     'scanpath')   
        ## For aoi sequence features 
        if True: 
            print('Segmenting aoi features...')
            aoi_feature_records = [feature_record for feature_record in self.feature_records
                                   if feature_record.split('.')[0].split('_')[-1] == 'AoI']
            self.signal_segmentation(aoi_feature_records, 
                                     'AoI')  
            
        
    def signal_segmentation(self, 
                            feature_records, type_,
                            display=False):
        '''
        

        Parameters
        ----------
        feature_records : TYPE
            DESCRIPTION.
        type_ : TYPE
            DESCRIPTION.
        display : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        '''
        if type_=='oculomotor':
            nb_bkps = self.config['symbolization']['segmentation']['oculomotor']['nb_breakpoints']
        elif type_=='scanpath':
            nb_bkps = self.config['symbolization']['segmentation']['scanpath']['nb_breakpoints']
        elif type_=='AoI':
            nb_bkps = self.config['symbolization']['segmentation']['aoi']['nb_breakpoints']
        outpath = 'output/results/ADABase/segmentation/'
        
        for record in feature_records: 
            subject, study, phase, level, _ = record.split('.')[0].split('_')
            label = '_'.join([study, phase, level])
            if subject in self.config['data']['subject_set'] and label in self.config['data']['label_set']:
                df = pd.read_csv(self.path+record)  
                name = record.split('.')[0] 
                
                ## Full signal segmentation
                signal = df.to_numpy()[:,1:] 
                if nb_bkps is not None: 
                    algo = rpt.KernelCPD(kernel="linear", jump=1).fit(signal)
                    bkps = algo.predict(n_bkps=nb_bkps) 
                else:
                    pen = np.log(signal.shape[0])/10
                    model='l2' # "l1", "rbf"
                    algo = rpt.Pelt(model=model, jump=1).fit(signal)
                    bkps = algo.predict(pen=pen) 
                ## Display segmentation if needed
                if display:
                    self.display_segmentation(signal, bkps, 
                                              name)
                bkps.insert(0, 0)
                filename = '{out_}{name_}.npy'.format(out_=outpath, 
                                                      name_=name)
                np.save(filename, np.array(bkps))
        
        
    def display_segmentation(self, 
                             signal, my_bkps, 
                             name=None):
        '''
        
    
        Parameters
        ----------
        signal : TYPE
            DESCRIPTION.
        my_bkps : TYPE
            DESCRIPTION.
        name : TYPE, optional
            DESCRIPTION. The default is None.
    
        Returns
        -------
        None.
    
        '''
        
        plt.style.use("seaborn-v0_8")  
        
        plt.plot(signal)
        for x in my_bkps[:-1]:
            plt.axvline(x = x-1, color = 'indianred', 
                        linewidth=5, linestyle='dashed')
        if name is not None:
            plt.title(name)
        plt.show()
        plt.clf()
      
        fig = plt.figure()
        ax = fig.add_subplot(111)  
        ax.imshow(signal.T, aspect=2.5, cmap='viridis', vmin=0, vmax=1)
        ax.grid(None)
        plt.show()
        plt.clf()
        
        fig = plt.figure()
        ax = fig.add_subplot(111) 
         
        ax.imshow(signal.T, aspect=2.5, cmap='viridis', vmin=0, vmax=1)
        ax.grid(None)
        
        for x in my_bkps[:-1]:
            ax.axvline(x = x-.5, color = 'indianred', 
                       linewidth=5, linestyle='dashed')
        if name is not None: 
            plt.title(name)
              
        plt.show()
        plt.clf()
                
                
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            