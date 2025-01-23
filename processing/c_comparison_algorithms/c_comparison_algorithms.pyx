# distutils: language = c++

from libcpp.map cimport map
from libcpp.pair cimport pair as cpair 
from libcpp cimport bool
 
import numpy as np


 


def generalized_edit (list s_1, list s_2, 
                      double del_c, double ins_c,
                      dict dict_chr_idx, double[:,:] dist_mat) :
    
    tmp_1, tmp_2, n_dict = int_convert_from_dict(s_1, s_2,
                                                 dict_chr_idx)
    
    cdef int[:] s_1_c = tmp_1
    cdef int[:] s_2_c = tmp_2
 
    ## Convert python dict to c dict for nogil loops
    cdef map[int, int] c_dict = dict_to_cmap(n_dict)
    
    ## Declare some constants
    cdef int n_1 = len(s_1)
    cdef int n_2 = len(s_2)
    
    cdef int i_ = 0
    cdef int j_ = 0 
    
    ## Declare the D matrix
    cdef double[:,:] d_mat = np.zeros((n_1+1, n_2+1), dtype=np.double)
 
    ## Fill the first column and row
    with nogil: 
        for i_ in range(n_1+1):
            d_mat[i_][0] = i_ * del_c 
        for j_ in range(n_2+1):
            d_mat[0][j_] = j_ * ins_c 
 
    cdef double w_s = 0.0
    cdef double w_d = 0.0
    cdef double w_i = 0.0
  
    cdef int s_s_0 = 0
    cdef int s_s_1 = 0
    
    cdef int i__ = 0
    cdef int j__ = 0 
    
    ## Fill the D-matrix
    with nogil: 
        for i__ in range (1, n_1+1):   
            for j__ in range (1, n_2+1):  
                w_s = d_mat[i__-1, j__-1] + dist_mat[c_dict[s_1_c[i__-1]],
                                                     c_dict[s_2_c[j__-1]]]
                 
                w_d = d_mat[i__-1, j__] + del_c 
                if w_d < w_s: 
                    w_s = w_d 
                 
                w_i = d_mat[i__, j__-1] + ins_c 
                if w_i < w_s: 
                    w_s = w_i 
                 
                d_mat[i__, j__] = w_s
               
 
     
    cdef double wf_s = d_mat[n_1, n_2] 
  
    return wf_s

 

cdef map[int, int] dict_to_cmap(dict p_dict):
    
    cdef int map_key
    cdef int map_val
    
    cdef cpair[int, int] map_e 
    cdef map[int, int] c_map
    
    for key,val in p_dict.items(): 
        map_key = key
        map_val = val   
        map_e = (map_key, map_val)
        c_map.insert(map_e)
        
    return c_map
 

def int_convert_from_dict(list s_1, list s_2,
                          dict dict_chr_idx):
    
    ## Create dict from str to int
    dict_str_int = dict()
    
    for i, str_ in enumerate(dict_chr_idx.keys()): 
        dict_str_int.update({str_: i})
     
    ## Convert input lists of str to list of int
    tmp_1 = np.array(
        [dict_str_int[s_1[i]] for i in range(len(s_1))], 
        dtype=np.int32
            ) 
    tmp_2 = np.array(
        [dict_str_int[s_2[j]] for j in range(len(s_2))], 
        dtype=np.int32
            ) 
    ## Create dict from int to indexes for the dist_mat
    n_dict = dict() 
    for key, val in dict_chr_idx.items(): 
        n_key = dict_str_int[key]
        n_dict.update({n_key: val})
        
    return tmp_1, tmp_2, n_dict
     