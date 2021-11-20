#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 01:46:08 2021

@author: ufuk
"""

import os
import argparse
import numpy as np
import cv2
import torch
import time

def mnn_ratio_matcher(descriptors1, descriptors2, ratio=0.8, bidirectional = True):
    
    # Mutual NN + symmetric Lowe's ratio test matcher for L2 normalized descriptors.
    device = descriptors1.device
    sim = descriptors1 @ descriptors2.t()

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = 2 - 2 * nns_sim
    # Compute Lowe's ratio.
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN and match similarity.
    nn12 = nns[:, 0]
    match_sim = nns_sim[:, 0]

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
    nns_dist = 2 - 2 * nns_sim
    # Compute Lowe's ratio.
    ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn21 = nns[:, 0]
    
    # if not bidirectional, do not use ratios from 2 to 1
    ratios21[:] *= 1 if bidirectional else 0
        
    # Mutual NN + symmetric ratio test.
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = torch.min(ids1 == nn21[nn12], torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio)) # discard ratios21 to get the same results with matlab
    # Final matches.
    matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)
    match_sim = match_sim[mask]

    return (matches.data.cpu().numpy(),match_sim.data.cpu().numpy())
    
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
    

    kaze = cv2.KAZE_create()
    
    with open(args_util.input_pairs) as f:
        start_time = time.time()
        for total_pair_number, line in enumerate(f):
            pairs = line.split(' ')
            p1_path = args_util.input_dir + '/' + pairs[0]
            p2_path = args_util.input_dir + '/' + pairs[1]
            
            img_A = cv2.imread(p1_path)
            img_B = cv2.imread(p2_path)
            
            keypoints0, descriptors0 = kaze.detectAndCompute(cv2.cvtColor(img_A,cv2.COLOR_BGR2GRAY),None)
            keypoints1, descriptors1 = kaze.detectAndCompute(cv2.cvtColor(img_B,cv2.COLOR_BGR2GRAY),None)
            
            #Arrange dtype
            descriptors0 = descriptors0.astype(float)
            descriptors1 = descriptors1.astype(float)
            
            #Normalize descs
            descriptors0 = torch.as_tensor(descriptors0 / np.sqrt((descriptors0*descriptors0).sum(axis=1))[:, np.newaxis])
            descriptors1 = torch.as_tensor(descriptors1 / np.sqrt((descriptors1*descriptors1).sum(axis=1))[:, np.newaxis])
            
            #if torch.cuda.is_available():
            #    descriptors0 = descriptors0.to('cuda')
            #    descriptors1 = descriptors1.to('cuda')
              
            #Find matches
            mtchs = mnn_ratio_matcher(descriptors0, descriptors1, 
                              ratio=0.9, bidirectional = True)

            mtchs = mtchs[0]
            
            #Remove GPU memory
            #torch.cuda.empty_cache()
            
            #Convert kpts to numpy array
            keypoints0 = cv2.KeyPoint_convert(keypoints0)
            keypoints1 = cv2.KeyPoint_convert(keypoints1)
            
            p1 = pairs[0].split('/')[2].split('.')[0]
            p2 = pairs[1].split('/')[2].split('.')[0]
                        
            np.savez_compressed(args_util.output_dir + '/' + p1 + '_' + p2 + '_' + 'matches', 
                                keypoints0=keypoints0, keypoints1=keypoints1, matches=mtchs)
                                
        end_time = time.time()  
        total_time = end_time - start_time
        avg_time = total_time / (total_pair_number+1)
           
        print(f'Total Execution Time for KAZE is: {total_time}')
        print(f'Average Execution Time for KAZE is: {avg_time}') 
