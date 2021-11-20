#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:15:00 2021

@authors: kutalmisince and ufukefe
"""
import os
import argparse
import numpy as np
import time

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
    
    #Since there is no matcher time start and stop timer here
    
    start_time = time.time()
    
    # Run SuperGlue, hiding output print with > /dev/null
    os.system('conda run -n ' + args.alg_name + ' python3 ' + args.alg_dir + '/' + 'match_pairs.py' +
              ' --input_dir ' + args.dataset_dir + ' --input_pairs ' + args.dataset_dir + '/' + 'image_pairs.txt' +
              ' --output_dir ' + args.output_dir + '/' + 'original_outputs' + ' --resize -1' + ' --match_threshold 0.9' ' --superglue "outdoor" > /dev/null')
              
    end_time = time.time()
    
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
                    mtchs = pair_out['matches']
                    match_confidence = pair_out['match_confidence']
                    
                    # Wrapper's Output
                    pointsA = keypoints0
                    pointsB = keypoints1
                    matches = np.vstack(((mtchs > -1).nonzero(), mtchs[mtchs > -1])).T.astype('int32')                  
                    
                    np.savez_compressed(args.output_dir + '/' + 'outputs' + '/' + subset + '/' + subsubset + 
                                        '/' + k, pointsA=pointsA, pointsB=pointsB, matches=matches)
                                       
    total_time = end_time - start_time
    avg_time = total_time / (total_pair_number+1)
                           
    print(f'Total Execution Time for SuperGlue is: {total_time}')
    print(f'Average Execution Time for SuperGlue is: {avg_time}') 
