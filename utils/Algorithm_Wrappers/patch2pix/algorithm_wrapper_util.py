#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:15:00 2021

@authors: kutalmisince and ufukefe
"""
import os
import argparse
from argparse import Namespace
from utils.eval.model_helper import init_ncn_matcher, init_patch2pix_matcher
import torch
import numpy as np
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Algorithm wrapper for Image Matching Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--alg_dir', type=str)
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--input_pairs', type=str)
    parser.add_argument('--output_dir', type=str) 
    parser.add_argument('--resize', type=str)    

    args_util = parser.parse_args()                    
    
    if not os.path.exists(args_util.output_dir):
        os.makedirs(args_util.output_dir)
        
    #Borrowed from Patch2Pix's ipnyb      
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    METHOD = 'patch2pix'
    
    if METHOD == 'nc':
        # Initialize ncnet matcher
        args = Namespace(ncn_thres=0.3, imsize=1024, ksize=2,
                          ckpt= args_util.alg_dir + '/pretrained/ncn_ivd_5ep.pth')
        matcher = init_ncn_matcher(args)
    else:
        # Initialize patch2pix matcher
        args = Namespace(io_thres=0.9, imsize=1024, ksize=2,
                          ckpt= args_util.alg_dir + '/pretrained/patch2pix_pretrained.pth')
        matcher = init_patch2pix_matcher(args)
    
    with open(args_util.input_pairs) as f:
        start_time = time.time()
        for total_pair_number, line in enumerate(f):
            pairs = line.split(' ')
            p1_path = args_util.input_dir + '/' + pairs[0]
            p2_path = args_util.input_dir + '/' + pairs[1]
                
            matches, _, _ = matcher(p1_path, p2_path)
            
            keypoints0 = matches[:, 0:2]
            keypoints1 = matches[:, 2:4]
            
            mtchs = np.vstack([np.arange(0,matches.shape[0])]*2).T
            
            p1 = pairs[0].split('/')[2].split('.')[0]
            p2 = pairs[1].split('/')[2].split('.')[0]
                        
            np.savez_compressed(args_util.output_dir + '/' + p1 + '_' + p2 + '_' + 'matches', 
                                keypoints0=keypoints0, keypoints1=keypoints1, matches=mtchs)
            torch.cuda.empty_cache()
            
        end_time = time.time()  
        total_time = end_time - start_time
        avg_time = total_time / (total_pair_number+1)
           
        print(f'Total Execution Time for Patch2Pix is: {total_time}')
        print(f'Average Execution Time for Patch2Pix is: {avg_time}') 

    
