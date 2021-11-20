#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:15:00 2021

@authors: kutalmisince and ufukefe
"""
import os
import argparse
import numpy as np
import torch
import time

#SuperPoint's Matcher

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


#First, extract and save the original algorithm's output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Algorithm wrapper for Image Matching Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--alg_name', type=str)
    parser.add_argument('--alg_dir', type=str)
    parser.add_argument('--dataset_dir', type=str) 
    parser.add_argument('--output_dir', type=str)    

    args = parser.parse_args()
    
    #Start time before feature extraction
    start_time = time.time()
    
    #Run SuperPoint, hiding output print with > /dev/null
    os.system('conda run -n ' + args.alg_name + ' python3 ' + args.alg_dir + '/' + 'match_pairs_sp.py' +
              ' --input_dir ' + args.dataset_dir + ' --input_pairs ' + args.dataset_dir + '/' + 'image_pairs.txt' +
              ' --output_dir ' + args.output_dir + '/' + 'original_outputs' + ' --resize -1' + ' --keypoint_threshold 0.005 > /dev/null') 

    #Then, read saved outputs and transform to proper format (keypointsA, keypointsB, matches)  
    pairs_out = os.listdir(args.output_dir  + '/' + 'original_outputs')
    
    with open(args.dataset_dir + '/' + 'image_pairs.txt') as f:
        for total_pair_number, line in enumerate(f):
            pairs = line.split(' ')
            subset = pairs[0].split('/')[0]
            subsubset = pairs[0].split('/')[1]
            p1 = pairs[0].split('/')[2].split('.')[0]
            p2 = pairs[1].split('/')[2].split('.')[0]
            
            if not os.path.exists(args.output_dir + '/' + 'outputs' + '/' + subset + '/' + subsubset):
                os.makedirs(args.output_dir + '/' + 'outputs' + '/' + subset + '/' + subsubset)
            
            for k in pairs_out:
                if p1 in k and p2 in k:
                    
                    # Original Algorithm's Output
                    pair_out = np.load(args.output_dir + '/' + 'original_outputs' + '/' + k)
                    
                    keypoints0 = pair_out['keypoints0']
                    keypoints1 = pair_out['keypoints1']
                    descriptors0 = pair_out['descriptors0']
                    descriptors1 = pair_out['descriptors1']
                    descriptors0 = descriptors0.T
                    descriptors1 = descriptors1.T
                      
                    #Arrange dtype
                    descriptors0 = descriptors0.astype(float)
                    descriptors1 = descriptors1.astype(float)

                    #Normalize descs
                    descriptors0 = torch.as_tensor(descriptors0 / np.sqrt((descriptors0*descriptors0).sum(axis=1))[:, np.newaxis])
                    descriptors1 = torch.as_tensor(descriptors1 / np.sqrt((descriptors1*descriptors1).sum(axis=1))[:, np.newaxis])
                            		    
                    if torch.cuda.is_available():
                        descriptors0 = descriptors0.to('cuda')
                        descriptors1 = descriptors1.to('cuda')
		    
                    #Find Matches (used .t() for tensor transpose different from classicals)
                    mtchs = mnn_ratio_matcher(descriptors0, descriptors1, 
                                              ratio=0.5, bidirectional = True)

                    mtchs = mtchs[0]
		    
                    #Remove GPU memory
                    torch.cuda.empty_cache()

                    # Wrapper's OutputS
                    pointsA = keypoints0
                    pointsB = keypoints1
                    matches = mtchs[0:2,:].T.astype('int32')                     
                    
                    np.savez_compressed(args.output_dir + '/' + 'outputs' + '/' + subset + '/' + subsubset + 
                                        '/' + k, pointsA=pointsA, pointsB=pointsB, matches=mtchs)
    
    end_time = time.time()                                     
    total_time = end_time - start_time
    avg_time = total_time / (total_pair_number+1)
                           
    print(f'Total Execution Time for SuperPoint is: {total_time}')
    print(f'Average Execution Time for SuperPoint is: {avg_time}') 
                    
          
