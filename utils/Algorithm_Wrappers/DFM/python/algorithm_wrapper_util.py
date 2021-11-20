#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:15:00 2021

@authors: kutalmisince and ufukefe
"""
import os
import argparse
from PIL import Image
import numpy as np
from DeepFeatureMatcher import DeepFeatureMatcher
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Algorithm wrapper for Image Matching Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--alg_dir', type=str)
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--input_pairs', type=str)
    parser.add_argument('--output_dir', type=str)   

    args_util = parser.parse_args()                    
    
    if not os.path.exists(args_util.output_dir):
        os.makedirs(args_util.output_dir)
    
    fm = DeepFeatureMatcher(enable_two_stage=True, model = 'VGG19_BN', 
                        ratio_th = [0.9, 0.9, 0.9, 0.9, 0.95, 1.0], bidirectional=True)
    
    with open(args_util.input_pairs) as f:
        start_time = time.time()
        for total_pair_number, line in enumerate(f):
            pairs = line.split(' ')
            p1_path = args_util.input_dir + '/' + pairs[0]
            p2_path = args_util.input_dir + '/' + pairs[1]
            
            img_A = np.array(Image.open(p1_path))
            img_B = np.array(Image.open(p2_path))
            
            H, H_init, points_A, points_B = fm.match(img_A, img_B)
            
            keypoints0 = points_A.T
            keypoints1 = points_B.T
        
            mtchs = np.vstack([np.arange(0,keypoints0.shape[0])]*2).T
            
            p1 = pairs[0].split('/')[2].split('.')[0]
            p2 = pairs[1].split('/')[2].split('.')[0]
                        
            np.savez_compressed(args_util.output_dir + '/' + p1 + '_' + p2 + '_' + 'matches', 
                                keypoints0=keypoints0, keypoints1=keypoints1, matches=mtchs)
         
        end_time = time.time()  
        total_time = end_time - start_time
        avg_time = total_time / (total_pair_number+1)
                           
        print(f'Total Execution Time for DFM is: {total_time}')
        print(f'Average Execution Time for DFM is: {avg_time}') 
    
