#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:24:36 2021

@authors: ufukefe and kutalmisince
"""

def get_names():
    
    #Single
    algorithms = ['dfm']
    alg_directory = {'dfm': 'Algorithms/DFM/python'}
    
    
    ## Learning-based
    # algorithms = ['superpoint', 'superglue', 'patch2pix', 'dfm']
    # alg_directory = {'superpoint': 'Algorithms/SuperPoint',
    #                   'superglue': 'Algorithms/SuperGlue',
    #                   'patch2pix': 'Algorithms/patch2pix',
    #                   'dfm': 'Algorithms/DFM/python'}
    
    
    ## Classical
    # algorithms = ['sift', 'surf', 'orb', 'kaze', 'akaze']
    # alg_directory = {'sift': 'Algorithms/sift',
    # 		      'surf': 'Algorithms/surf',
    # 		      'orb': 'Algorithms/orb',
    # 		      'kaze': 'Algorithms/kaze',
    # 		      'akaze': 'Algorithms/akaze'}
    
    # #All
    # algorithms = ['sift', 'surf', 'orb', 'kaze', 'akaze', 'superpoint', 'superglue', 'patch2pix', 'dfm']
    # alg_directory = {'sift': 'Algorithms/sift',
    # 		      'surf': 'Algorithms/surf',
    # 		      'orb': 'Algorithms/orb',
    # 		      'kaze': 'Algorithms/kaze',
    # 		      'akaze': 'Algorithms/akaze', 
    #               'superpoint': 'Algorithms/SuperPoint',
    #               'superglue': 'Algorithms/SuperGlue',
    #               'patch2pix': 'Algorithms/patch2pix',
    #               'dfm': 'Algorithms/DFM/python'}
    
    #Single dataset
    datasets = ['hpatches']
    dataset_directory = {'hpatches': 'Datasets/hpatches'}
    
    ##All datasets     
    # datasets = ['multi_modal', 'hpatches']
    # dataset_directory = {'hpatches': 'Datasets/hpatches',
    #                      'multi_modal': 'Datasets/multi_modal'}
    
    result_directory = 'Results'

    return  algorithms, alg_directory, datasets, dataset_directory, result_directory
