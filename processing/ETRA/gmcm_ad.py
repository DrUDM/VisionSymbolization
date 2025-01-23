# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:41:27 2021

@author: marca
"""

import numpy as np 
import matplotlib.pyplot as plt
import sys
from scipy.linalg import sqrtm
from sklearn.cluster import KMeans 
from sklearn.mixture import GaussianMixture
import tensorflow as tf
import copy
from scipy.special import erf
import bisect 
from statsmodels.distributions.empirical_distribution import ECDF
from collections import Counter 
import time 
import torch

import tensorflow_probability as tfp
tfd = tfp.distributions

np.random.seed(50)



class AD_GMCM():
    
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
        self.master_stepzise = config['symbolization']['gmcm']['master_stepsize']
        self.nb_steps = config['symbolization']['gmcm']['nb_steps']
        
        ## Initialize some variables
        self.U = None
        self.uniformized = uniformized 
        
        self.mu = None 
        self.sigma_chol = None
        self.sigma = None
        
        self.alpha = None 
        self.pi = None 
         
        self.likelihoods = np.zeros(self.nb_steps)
        self.clusters = None
        
        self.beta_1 = config['ADAM']['beta_1']
        self.beta_2 = config['ADAM']['beta_2']
        
        ## For ADAM Optimizer
        self.m_ = dict({'alpha': np.zeros(K), 
                        'sigma_chol': dict({}), 
                        'mu': dict({})}) 
        self.v_ = dict({'alpha': np.zeros(K), 
                        'sigma_chol': dict({}), 
                        'mu': dict({})}) 
        
        
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
                              self.sigma, self.sigma_chol, self.mu, self.alpha, self.pi, 
                              self.master_stepzise, i+1, 
                              self.beta_1, self.beta_2, 
                              self.m_, self.v_) 
            local.process()
          
            ## Update moments for ADAM optimizer
            self.m_ = local.m_
            self.v_ = local.v_
            ## Update model parameters
            self.alpha = local.new_alpha
            self.pi = local.new_pi 
            self.mu =local.new_mu
            self.sigma_chol = local.new_sigma_chol
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
        print(Counter(self.clusters))
        
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
        
        for k in range(self.K):
            self.m_['sigma_chol'].update({k: np.zeros((self.D, self.D))})
            self.v_['sigma_chol'].update({k: np.zeros((self.D, self.D))})
            
            self.m_['mu'].update({k: np.zeros(self.D)})
            self.v_['mu'].update({k: np.zeros(self.D)})
            
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
        #GMM = GaussianMixture(n_components=self.K)
        #GMM.fit(self.U)
      
        centers = kmeans.cluster_centers_   
        mu_init, sigma_cholesky_init, sigma_init=dict({}), dict({}), dict({})
        
        for k in range(self.K):
            mu_init.update({k: centers[k]}) 
            #mu_init.update({k: GMM_mu[k]}) 
            sigma_cholesky_init.update({k: .1 * np.identity(self.D)})
            sigma_init.update({k: (sigma_cholesky_init[k] 
                                   @ sigma_cholesky_init[k].T)})  
            #sigma_cholesky_init.update({k: np.linalg.cholesky(GMM_sigmas[k])})
            #sigma_init.update({k: GMM_sigmas[k]}) 
            
        self.mu, self.sigma, self.sigma_chol = mu_init, sigma_init, sigma_cholesky_init
        alpha_init = np.ones(self.K) 
        pi_init = np.exp(alpha_init)/np.sum(np.exp(alpha_init)) 
        self.alpha, self.pi = alpha_init, pi_init 
       
        
    def predict(self, X, 
                uniformized=False):
        '''
        

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        uniformized : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        N, D = X.shape
        assert D == self.D
        
        if uniformized: 
            U = X 
        else: 
            U = np.zeros((N, D)) 
            for d in range (D):
                datas = X[:,d]
                ecdf = ECDF(datas)
                u = ecdf(datas) 
                U[:,d] = u    
             
        CDFi = CDF_Inversion (U, self.K, 
                              self.sigma, self.mu, self.pi,
                              nb_evaluations=500, disp_intervals=4)   
        CDFi.process()   
        Z = CDFi.Z
         
        pi = self.pi 
        mu = self.mu
        V_ = self.sigma_chol
     
        num = np.zeros((self.K, N))
        denom = np.zeros((D, N))
    
        for k in range (self.K): 
            sigma = np.matmul(V_[k], np.transpose(V_[k]))   
            sigma_inv = np.linalg.inv(sigma)
            sigma_det = np.linalg.det(sigma)
        
            Z_ctrd = Z - mu[k] 
            local_num = -0.5*np.diag(np.matmul(np.matmul(Z_ctrd, 
                                                         sigma_inv),
                                               Z_ctrd.T))
            num[k] = (pi[k]
                      * pow(((2*np.pi)**(D)) * sigma_det, -1/2) 
                      * np.exp(local_num))
         
            local_denom = (-0.5*np.matmul(np.diag( 1/(np.diag(sigma))), 
                                          (Z_ctrd.T)**2))                     
            denom = denom + np.matmul(pi[k]*np.diag((2*np.pi*np.diag(sigma))**(-1/2)), 
                                        np.exp(local_denom))    
        denom = np.prod(denom, axis = 0)
        local_post_probs = num/denom
        total_probs = np.sum(local_post_probs, axis = 0)
    
        local_cond_probs = local_post_probs/total_probs
        clusters = []
        
        for i in range(N):
            clust = np.argmax(local_cond_probs[:,i])
            clusters.append(clust)
            
        return np.array(clusters)


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
                 sigma, sigma_chol, mu, alpha, pi,
                 master_stepzise, iteration,
                 beta_1, beta_2, 
                 m_, v_):
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
        
        self.N, self.D = input.shape 
        self.K = K        
        ## Model variables
        self.mu = mu
        self.new_mu = dict({})       
        self.sigma_chol, self.sigma = sigma_chol, sigma 
        self.new_sigma_chol, self.new_sigma = dict({}), dict({}) 
        self.alpha, self.pi = alpha, pi 
        self.new_alpha, self.new_pi = None, None
        ## Variables and parameters for ADAM optimizer
        self.master_stepzise = master_stepzise  
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m_ = m_ 
        self.v_ = v_
        self.iteration = iteration
        self.eps = 1e-10
        ## Local gradients
        self.mu_grad, self.sigma_chol_grad = dict({}), dict({}) 
        self.alpha_grad = None
       
    
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
    
   
    def process_gradients (self):
        '''
        

        Returns
        -------
        None.

        '''
        '''
        start_time = time.time()
         
        Z_to = torch.tensor(self.Z, 
                            requires_grad=False)
        pi_scal_to = torch.tensor(np.pi, 
                                  dtype=torch.float32, 
                                  requires_grad=False)
        pi_pow_scal_to = torch.tensor((2*np.pi)**(self.D), 
                                      dtype=torch.float32, 
                                      requires_grad=False)
        alpha_to = torch.tensor(self.alpha, 
                                requires_grad=True)
        pi_to = torch.exp(alpha_to)/torch.sum(torch.exp(alpha_to))
        num_, mu_to, V_to= dict(), dict(), dict()
        for k in range (self.K):       
            mu_to[k] = torch.tensor(self.mu[k], 
                                    requires_grad=True)
            V_to[k] = torch.tensor(self.sigma_chol[k], 
                                   requires_grad=True)
            
            sigma_to = torch.matmul(V_to[k], torch.transpose(V_to[k], 0, 1))   
            sigma_inv_to = torch.linalg.inv(sigma_to)
            sigma_det_to = torch.linalg.det(sigma_to)
        
            Z_to_ctrd = Z_to - mu_to[k]  
            ## Num
            local_num = -0.5*torch.diag(torch.matmul(torch.matmul(Z_to_ctrd, 
                                                                  sigma_inv_to),
                                                     torch.transpose(Z_to_ctrd, 0, 1)), 0)
            num_[k] = (pi_to[k]
                       * torch.pow(pi_pow_scal_to * sigma_det_to, -1/2) 
                       * torch.exp(local_num)) 
            ## Denom
            local_denom = (-0.5*torch.matmul(torch.diag(torch.pow(torch.diag(sigma_to), 
                                                                        -1) ), 
                                                 torch.pow(torch.transpose(Z_to_ctrd, 0, 1),2)))
            if k==0:
                denom_ = (torch.matmul(pi_to[k]*torch.diag(torch.pow(2*pi_scal_to*torch.diag(sigma_to),
                                                                     -1/2) ), 
                                       torch.exp(local_denom))) 
            else:
                denom_ = denom_ + (torch.matmul(pi_to[k]*torch.diag(torch.pow(2*pi_scal_to*torch.diag(sigma_to),
                                                                              -1/2) ), 
                                                torch.exp(local_denom))) 
        num_sum = num_[0] 
        for k in range (1, self.K):
            num_sum = num_sum + num_[k]
        num_sum_log = torch.log(num_sum)
        denom_ = torch.sum(torch.log(denom_), dim=0)  
        vect_log_likelihood = num_sum_log - denom_ 
        
        log_likelihood = torch.sum(vect_log_likelihood) 
        log_likelihood.backward()
 
            
        print("--- %s seconds ---" % (time.time() - start_time))  
        '''
        #start_time = time.time()
        with tf.GradientTape(persistent = False) as g:    
            Z_tf = tf.constant(self.Z)
      
            pi_scal_tf = tf.constant(np.pi, dtype='float64')
            #pi_pow_scal_tf = tf.constant((2*np.pi)**(self.D), dtype='float64')
        
            alpha_tf = tf.Variable(self.alpha)
            pi_tf = tf.math.exp(alpha_tf)/tf.reduce_sum(tf.math.exp(alpha_tf))
        
            Z_tf_ctrds, sigmas_tf, num_, mu_tf, V_tf= dict(), dict(), dict(), dict(), dict()
            for k in range (self.K):       
                mu_tf[k] = tf.Variable(self.mu[k])
                V_tf[k] = tf.Variable(self.sigma_chol[k])
      
                sigma_tf = tf.linalg.matmul(V_tf[k], tf.transpose(V_tf[k])) 
                sigmas_tf[k] = sigma_tf
                #sigma_inv_tf = tf.linalg.inv(sigma_tf)
                #sigma_det_tf = tf.linalg.det(sigma_tf)
            
                Z_tf_ctrd = Z_tf - mu_tf[k] 
                Z_tf_ctrds[k] = Z_tf_ctrd
                ## Num
                #local_num = -0.5*tf.linalg.diag_part(tf.linalg.matmul(tf.linalg.matmul(Z_tf_ctrd, 
                #                                                                       sigma_inv_tf),
                #                                                      tf.transpose(Z_tf_ctrd)))
                
                #num_[k] = (pi_tf[k]
                #           * tf.pow(pi_pow_scal_tf * sigma_det_tf, -1/2) 
                #           * tf.math.exp(local_num)) 
                num_[k] = pi_tf[k] * tfd.MultivariateNormalTriL(
                                                        loc=mu_tf[k],
                                                        scale_tril=tf.linalg.cholesky(sigma_tf)).prob(Z_tf)
                ## Denom
                local_denom = (-0.5*tf.linalg.matmul(tf.linalg.diag(tf.pow(tf.linalg.diag_part(sigma_tf), 
                                                                            -1) ), 
                                                     tf.pow(tf.transpose(Z_tf_ctrd),2)))
                if k==0:
                    denom_ = (tf.linalg.matmul(pi_tf[k]*tf.linalg.diag(tf.pow(2*pi_scal_tf*tf.linalg.diag_part(sigma_tf),
                                                                                       -1/2) ), 
                                                      tf.math.exp(local_denom))) 
                else:                                           
                    denom_ = denom_ + (tf.linalg.matmul(pi_tf[k]*tf.linalg.diag(tf.pow(2*pi_scal_tf*tf.linalg.diag_part(sigma_tf),
                                                                                   -1/2) ), 
                                                  tf.math.exp(local_denom))) 
           
            #for j in range(self.D): 
            #    for k in range(self.K): 
            #        sigma_tf = sigmas_tf[k]
            #        
            #        if k==0: 
            #            denom_n = pi_tf[k] *  tfd.Normal(loc=mu_tf[k][j], scale=tf.math.sqrt(sigma_tf[j,j])).prob(Z_tf[:,j])
            #        else:  
            #            denom_n = denom_n + pi_tf[k] * tfd.Normal(loc=mu_tf[k][j], scale=tf.math.sqrt(sigma_tf[j,j])).prob(Z_tf[:,j])
            #    if j ==0:
            #        denom_m = denom_n
            #    else:
            #        denom_m = denom_m * denom_n
            #denom_m = tf.math.log(denom_m)
            
         
            #for j in range(self.D):  
            #    denom_n = tf.reshape(pi_tf, [1, self.K]) * tfd.Normal(loc=[mu_tf[k][j] for k in range(self.K)], 
            #                                             scale=[tf.math.sqrt(sigmas_tf[k][j,j]) for k in range(self.K)]).prob(tf.reshape(Z_tf[:,j], [self.N, 1]))
            #    denom_n = tf.reduce_sum(denom_n, axis=1)
            #    if j==0:
            #        denom_m = denom_n
            #    else:
            #        denom_m = denom_m*denom_n 
            #denom_m = tf.math.log(denom_m)
          
            num_sum = num_[0] 
            for k in range (1, self.K):
                num_sum = num_sum + num_[k]
                
            num_sum_log = tf.math.log(num_sum)
            denom_ = tf.math.reduce_sum(tf.math.log(denom_), axis=0) 
            vect_log_likelihood = num_sum_log - denom_  
            #vect_log_likelihood = num_sum_log - denom_m
            
            log_likelihood = tf.math.reduce_sum(vect_log_likelihood) 
             
            likelihood_ar = np.array(log_likelihood) 
            self.likelihood = copy.deepcopy(likelihood_ar)  
            ## Return hard copied grads 
            grad = g.gradient(log_likelihood, [alpha_tf, 
                                               mu_tf.values(), 
                                               V_tf.values()])  
            #print("--- %s seconds ---" % (time.time() - start_time))   
            grad_mu = np.array(grad[1])
            grad_V = np.array(grad[2])  
            for k in range (self.K):
                self.mu_grad[k] = copy.deepcopy(grad_mu[k])
                self.sigma_chol_grad[k] = copy.deepcopy(grad_V[k]) 
            self.alpha_grad = copy.deepcopy(np.array(grad[0]))
            
         
    
 
    def adam_opt(self, gradient, 
                 m_, v_):
        '''
        

        Parameters
        ----------
        gradient : TYPE
            DESCRIPTION.
        m_ : TYPE
            DESCRIPTION.
        v_ : TYPE
            DESCRIPTION.

        Returns
        -------
        m_ : TYPE
            DESCRIPTION.
        v_ : TYPE
            DESCRIPTION.
        m_h : TYPE
            DESCRIPTION.
        v_h : TYPE
            DESCRIPTION.

        '''
        m_ = self.beta_1*m_ + (1-self.beta_1)*gradient
        v_ = self.beta_2*v_ + (1-self.beta_2)*(gradient**2)
        
        m_h = m_ / (1 - self.beta_1**self.iteration)
        v_h = v_ / (1 - self.beta_2**self.iteration)
        
        return m_, v_, m_h, v_h
        
 
    def update_params (self):
        '''
        

        Returns
        -------
        None.

        '''
        var_eps = self.master_stepzise 
        for k in range(self.K):
            ## Update mu
            m_, v_, m_h, v_h = self.adam_opt(self.mu_grad[k] , 
                                             self.m_['mu'][k], 
                                             self.v_['mu'][k])
            self.m_['mu'][k]=m_
            self.v_['mu'][k]=v_
            self.new_mu[k] = self.mu[k] + (var_eps * m_h)/(np.sqrt(v_h)+self.eps)
            ## Update sigma_chol
            m_, v_, m_h, v_h = self.adam_opt(self.sigma_chol_grad[k] , 
                                             self.m_['sigma_chol'][k], 
                                             self.v_['sigma_chol'][k])
            self.m_['sigma_chol'][k]=m_
            self.v_['sigma_chol'][k]=v_
            new_sigma_chol_local = self.sigma_chol[k] + (var_eps * m_h)/(np.sqrt(v_h)+self.eps)
            self.new_sigma_chol[k] = new_sigma_chol_local
            self.new_sigma[k] = np.matmul(new_sigma_chol_local, 
                                          new_sigma_chol_local.T)  
        ## Update alpha
        m_, v_, m_h, v_h = self.adam_opt(self.alpha_grad , 
                                         self.m_['alpha'],
                                         self.v_['alpha'])
        self.m_['alpha']=m_
        self.v_['alpha']=v_
        new_alpha = self.alpha + (var_eps * m_h)/(np.sqrt(v_h)+self.eps)    
        self.new_alpha = new_alpha
        self.new_pi = np.exp(new_alpha)/np.sum(np.exp(new_alpha))  
         
        
    def process (self):
        '''
        

        Returns
        -------
        None.

        '''
    
        
        self.reset_latent_variable()  
        self.process_gradients()  
        self.update_params()
        
            
   
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
 



