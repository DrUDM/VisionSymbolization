# -*- coding: utf-8 -*-



import numpy as np 
import matplotlib.pyplot as plt
import sys
from scipy.linalg import sqrtm
from sklearn.cluster import KMeans 
import tensorflow as tf
import copy
from scipy.special import erf
import bisect 
from statsmodels.distributions.empirical_distribution import ECDF
 
from scipy.stats import multivariate_normal as mvn
import time

## From "Parametric Characterization of Multimodal Distributions with Non-gaussian Modes"
class PEM_GMCM():
    
    def __init__(self,
                 K, config, 
                 uniformized=False):
        '''
        

        Parameters
        ----------
        input : TYPE
            DESCRIPTION.
        K : TYPE
            DESCRIPTION.
        config : TYPE
            DESCRIPTION.
        uniformized : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        ''' 
        self.likelihood = None
        self.continu = True
        self.distance = config['symbolization']['gmcm']['distance']
        ## X with dimension N x D
        self.X = None 
        self.N, self.D = None, None
        ## Number of cluster
        self.K = K   
        
        ## Model parameters
        self.nb_steps = config['symbolization']['gmcm']['nb_steps']
        
        ## Initialize some variables
        self.U = None
        self.uniformized = uniformized 
        
        self.mu = None  
        self.sigma = None
        
        self.alpha = None 
        self.pi = None 
         
        self.likelihoods = np.zeros(self.nb_steps)
        self.clusters = None
         
        
    def fit(self, input):
        '''
        

        Returns
        -------
        None.

        '''
        ## Initialize parameters and compte uniformized variables if necessary
        self.initialize(input)
      
        for i in range (0, self.nb_steps): 
            local = LocalStep(self.U, self.K,
                              self.sigma, self.mu, self.pi) 
            local.process()
           
            ## Update model parameters 
            self.pi = local.new_pi 
            self.mu =local.new_mu 
            self.sigma = local.new_sigma
            ## Update likelihood
            lk = local.likelihood
            self.likelihoods[i] = lk
            sys.stdout.write("\r Log likelihood: {likelihood} for step {i} over {tot}       ".format(likelihood=lk,
                                                                                                     i = i+1,
                                                                                                     tot = self.nb_steps))
            sys.stdout.flush()  
        print('\n') 
        self.clusters = self.predict(self.U, True)
        #print(Counter(self.clusters))
        
        fig, ax = plt.subplots()
        ax.plot(self.likelihoods)
        plt.show()
        plt.clf()
      
  
    def initialize(self, input):
        '''
        

        Returns
        -------
        None.

        '''
        ## Finish initialization according to input data
        self.X = input 
        self.N, self.D = input.shape  
           
        ## Compute uniformized variable
        if self.uniformized: 
            self.U = self.X 
        else: 
            U = np.zeros((self.N, self.D)) 
            for d in range (self.D):
                datas = self.X[:,d]
                ecdf = ECDF(datas)
                u = ecdf(datas) 
                U[:,d] = u    
            self.U = U 
        ## Compute K-mean centers to initialize mu
        kmeans = KMeans(n_clusters=self.K, 
                        n_init=100, 
                        random_state=0).fit(self.U)
        centers = kmeans.cluster_centers_   
        mu_init, sigma_init=dict(), dict() 
        
        for k in range(self.K):
            mu_init.update({k: centers[k]}) 
            #mu_init.update({k: np.random.random(self.D)}) 
            sigma_init.update({k: 1 * np.identity(self.D)})  
        self.mu, self.sigma = mu_init, sigma_init  
        self.pi = np.ones(self.K)*(1/self.K)
   
        
        
    def predict(self, X, 
                uniformized=False): 
        return 0


    def distance_matrix(self):
        
        mu, sigma = self.mu, self.sigma
        d_mat = np.zeros((self.K, self.K))
        
        for j in range(1, self.K): 
            for i in range(j):  
                if self.distance == 'wasserstein':
                    d_mat[i,j] = self.wasserstein(mu[i], mu[j], 
                                                  sigma[i], sigma[j])
                elif self.distance == 'jeffreys':
                    d_mat[i,j] = self.jeffreys(mu[i], mu[j], 
                                               sigma[i], sigma[j])
                elif self.distance == 'bhattacharyya':
                    d_mat[i,j] = self.bhattacharyya(mu[i], mu[j], 
                                               sigma[i], sigma[j])
                elif self.distance == 'naive':
                    d_mat[i,j] = np.linalg.norm(mu[i]-mu[j])
                d_mat[j,i] = d_mat[i,j]
        return d_mat
    
  
    def wasserstein(self, 
                    mu_1, mu_2, 
                    sigma_1, sigma_2):
        
        wd = np.linalg.norm(mu_1-mu_2)**2
        wd += np.trace(sigma_1) + np.trace(sigma_2)
        
        sigma_1_sqrt = sqrtm(sigma_1)
        wd -= 2*np.trace(sqrtm(sigma_1_sqrt @ sigma_2 @ sigma_1_sqrt))
        
        return np.sqrt(wd)
    
    
    def jeffreys(self, 
                 mu_1, mu_2, 
                 sigma_1, sigma_2):
        
        sigma_1_inv = np.linalg.inv(sigma_1)
        sigma_1_det = np.linalg.det(sigma_1)
        sigma_2_inv = np.linalg.inv(sigma_2)
        sigma_2_det = np.linalg.det(sigma_2)
        
        jd = 0.5*(np.log(sigma_2_det/sigma_1_det) - self.D 
                  + (mu_1-mu_2).T @ sigma_2_inv @ (mu_1-mu_2) 
                  + np.trace(sigma_2_inv @ sigma_1))
        jd += 0.5*(np.log(sigma_1_det/sigma_2_det) - self.D 
                   + (mu_2-mu_1).T @ sigma_1_inv @ (mu_2-mu_1) 
                   + np.trace(sigma_1_inv @ sigma_2))
        return jd

    
    def bhattacharyya(self, 
                 mu_1, mu_2, 
                 sigma_1, sigma_2):
        
        sigma = (sigma_1 + sigma_2)/2
        sigma_inv = np.linalg.inv(sigma)
        sigma_det = np.linalg.det(sigma) 
        
        sigma_1_det = np.linalg.det(sigma_1) 
        sigma_2_det = np.linalg.det(sigma_2)
        
        bd = (1/8)*((mu_1-mu_2).T @ sigma_inv @ (mu_1-mu_2))
        bd += .5*(np.log(sigma_det/(np.sqrt(sigma_1_det * sigma_2_det))))
        
        return bd 
    
    
class LocalStep():
    
    def __init__(self,
                 input, K,
                 sigma, mu, pi):
        '''
        

        Parameters
        ----------
        input : TYPE
            DESCRIPTION.
        K : TYPE
            DESCRIPTION.
        sigma : TYPE
            DESCRIPTION.
        sigma_chol : TYPE
            DESCRIPTION.
        mu : TYPE
            DESCRIPTION.
        alpha : TYPE
            DESCRIPTION.
        pi : TYPE
            DESCRIPTION.
        master_stepzise : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''  
        self.U = input
        self.Z = None
        self.likelihood = None
        self.h = None
        
        self.N, self.D = input.shape 
        self.K = K        
        ## Model variables 
        self.mu = mu
        self.new_mu = dict({})       
        self.sigma = sigma 
        self.new_sigma = dict()
        self.pi = pi 
        self.new_pi = None
        ## Variables and parameters for ADAM optimizer
       
        
    def reset_latent_variable (self):
        '''
        

        Returns
        -------
        None.

        '''
        CDFi = CDF_Inversion (self.U, self.K, 
                              self.sigma, self.mu, self.pi,
                              nb_evaluations=2000, disp_intervals=5)   
        CDFi.process()   
        self.Z = CDFi.Z
    
  
    def update_params (self): 
        
        h = np.zeros((self.N, self.K))
        Z = self.Z
        pi = self.pi 
        mu, sigma = self.mu, self.sigma
        ## E-Step
        for k in range(self.K): 
            dist = mvn(mean=mu[k], cov=sigma[k]) 
            h[:,k] = pi[k]* dist.pdf(Z)
        norm_ = np.sum(h, axis=1) 
        h /= norm_.reshape((self.N, 1))
        self.h = h
        
        ## M-Step
        new_pi = (np.sum(h, axis = 0)/self.N) 
        self.new_pi = new_pi/np.sum(new_pi)
        print(self.new_pi)
        for k in range(self.K): 
            ## Update mu
            mu = self.mu[k]
            sum_hk = np.sum(h[:,k]) 
            print(sum_hk)
            n_mu = np.sum(h[:,k].reshape((self.N, 1))*Z, axis=0)  
            n_mu /= sum_hk
            self.new_mu.update({k: n_mu})
              
            ## Update sigma
            Z_ctrd = Z-mu 
            n_sigma = np.sum(np.array(
                [Z_ctrd[i].reshape((self.D, 1)) @ Z_ctrd[i].reshape((1, self.D)) for i in range(self.N)]
                ), axis = 0)
            
            n_sigma /= sum_hk
            self.new_sigma.update({k: n_sigma })
   
        
    def comp_likelihood (self): 
        
        pi, mu, sigma = self.new_pi, self.new_mu, self.new_sigma
        Z = self.Z
        num = np.zeros((self.N, self.K))
        for k in range(self.K): 
            dist = mvn(mean=mu[k], cov=sigma[k]) 
            num[:,k] = pi[k]* dist.pdf(Z)
        num = np.sum(num, axis = 1)
        
        denom = np.zeros((self.N, self.D))
        for j in range(self.D): 
            l_d = np.array(
                [mvn(mean=mu[k][j], cov=sigma[k][j,j]).pdf(Z[:,j].reshape((self.N, 1))) for k in range(self.K)]
                )
            l_d = np.sum(l_d, axis=0)
            denom[:,j] = l_d
        denom = np.prod(denom, axis=1)
        
        log_likelihood = np.sum(np.log(num) - np.log(denom))
        self.likelihood = log_likelihood
        
        
    def process (self):
        '''
        

        Returns
        -------
        None.

        '''
        start_time = time.time()
        self.reset_latent_variable()  
        self.update_params()
        self.comp_likelihood()
        print("--- %s seconds ---" % (time.time() - start_time))
 

class CDF_Inversion():
    
    def __init__(self,
                 U, K,
                 sigma, mu, pi,
                 nb_evaluations, disp_intervals):
        '''
        

        Parameters
        ----------
        U : TYPE
            DESCRIPTION.
        K : TYPE
            DESCRIPTION.
        sigma : TYPE
            DESCRIPTION.
        mu : TYPE
            DESCRIPTION.
        pi : TYPE
            DESCRIPTION.
        nb_evaluations : TYPE
            DESCRIPTION.
        disp_intervals : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.U = U
        self.N, self.D = U.shape 
        
        self.Z = np.zeros((self.N, self.D))
        self.pi = pi
        self.K = K
         
        self.mu = mu
        self.sigma = sigma
         
        self.nb_evaluation_points = nb_evaluations
        self.nb_evals_to_use = None
        self.disp_intervals = disp_intervals
        
        self.repart = None
        self.evaluation_points = None 
        self.cumul_density_grid = None
        
 
    def process_evaluation_points (self):
         '''
        

        Returns
        -------
        None.

        '''
         repart = self.pi*self.nb_evaluation_points  
      
         self.nb_evals_to_use = int(np.round(repart).sum()) 
        
         evaluation_points = np.zeros((self.nb_evals_to_use, self.D))
         idx = 0
         for k in range (0, self.K):
             disp = self.disp_intervals*np.sqrt(np.diag(self.sigma[k]))
             evaluation_points[idx:idx+int(np.round(repart[k])), :] = np.linspace(self.mu[k] - disp, 
                                                                                  self.mu[k] + disp, 
                                                                                  int(np.round(repart[k])))
             idx += int(np.round(repart[k]))
   
         self.evaluation_points = evaluation_points

 
    def process_cumul_density_grid (self):
        '''
        

        Returns
        -------
        None.

        '''
        cumul_density_grid = np.zeros((self.nb_evals_to_use, self.D))
        for j in range (0, self.D):
            e_points_local = self.evaluation_points[:,j]  
            for k in range (0, self.K):
                loc_variable = (e_points_local - self.mu[k][j])/(np.sqrt(2*self.sigma[k][j,j]))
                cumul_density_grid[:,j] += self.pi[k] * 0.5 * (erf(loc_variable) + 1)
          
        self.cumul_density_grid = cumul_density_grid         
                
 
    def search_interpol (self):
        '''
        

        Returns
        -------
        None.

        '''
        u_grid = self.cumul_density_grid
        v_grid = self.evaluation_points
        U = self.U 
        max_idx = self.nb_evals_to_use - 1
        
        for j in range (0, self.D):
            local_u_grid = u_grid[:,j]
            local_v_grid = v_grid[:,j]
            local_U = U[:,j]   
            
            idx_v = np.array([bisect.bisect_left(local_u_grid, u) for u in local_U]) 
            idx_v = np.where(idx_v > max_idx, max_idx, idx_v)
            
            y_maj, y_min = local_u_grid[idx_v], local_u_grid[idx_v-1] 
            x_maj, x_min = local_v_grid[idx_v], local_v_grid[idx_v-1] 
              
            inverse_interpol = x_min + (local_U - y_min)*(x_maj - x_min)/(y_maj - y_min) 
            self.Z[:,j] = inverse_interpol 
    
        
    def process (self):
        '''
        

        Returns
        -------
        None.

        '''
        self.process_evaluation_points() 
        self.process_cumul_density_grid() 
        self.search_interpol() 
 


