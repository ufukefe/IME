#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:15:00 2021

@authors: kutalmisince and ufukefe
"""

import argparse
import numpy as np
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Algorithm wrapper for Image Matching Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--algorithms', '--algorithms', nargs='+', default=[])
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--result_directory', type=str)
    parser.add_argument('--dataset_dir', type=str) 

    args = parser.parse_args()
    algorithms = args.algorithms
    dataset = args.dataset_name
    result_directory = args.result_directory
    dataset_dir = args.dataset_dir
    
#For calculating Mean Matching Accuracy (MMA)
    
#Borrowed from https://github.com/GrumpyZhou/image-matching-toolbox
        
def eval_matches(p1s, p2s, homography):
    # Compute the reprojection errors from im1 to im2 
    # with the given the GT homography
    p1s_h = np.concatenate([p1s, np.ones([p1s.shape[0], 1])], axis=1)  # Homogenous
    p2s_proj_h = np.transpose(np.dot(homography, np.transpose(p1s_h)))
    p2s_proj = p2s_proj_h[:, :2] / p2s_proj_h[:, 2:]
    dist = np.sqrt(np.sum((p2s - p2s_proj) ** 2, axis=1))
    return dist        
        
alg_results = {}
    
for i, alg in enumerate(algorithms):
    out_dir = result_directory + '/' + dataset + '/' + alg
    
    all_results = np.empty(shape=[0, 12])
    subset_dict = {}
    subset_dict['all'] = {}
    
    #For all pairs in image_pairs.txt do evaluation
    with open(dataset_dir + '/' + 'image_pairs.txt') as f:
        for k, line in enumerate(f):
            pairs = line.split(' ')
            subset = pairs[0].split('/')[0]
            subsubset = pairs[0].split('/')[1]
                           
            p1 = pairs[0].split('/')[2].split('.')[0]
            p2 = pairs[1].split('/')[2].split('.')[0]
            
        
            #Load output points and matches
            outputs = np.load(out_dir + '/' + 'outputs' + '/' + subset + '/' + subsubset + '/' + 
                              p1 + '_' + p2 + '_' + 'matches' + '.npz')
    
            pointsA = outputs['pointsA']
            pointsB = outputs['pointsB']
            matches = outputs['matches']
            
            #Load groundtruth homographies

            h_name = pairs[2].split('/')[2].split('\n')[0]
            h_gt = np.loadtxt(dataset_dir + '/' + subset + '/' + subsubset + '/' + h_name)

            pointsA_matched = pointsA[matches[:,0]]
            pointsB_matched = pointsB[matches[:,1]]
            
            distances = eval_matches(pointsA_matched, pointsB_matched, h_gt)
            
            if distances.shape[0] >= 1:
                mma = np.around(np.array([np.count_nonzero(distances <= i)/distances.shape[0] 
                            for i in range (1,11)]),3)
            else:
                mma = np.zeros(10)
            
            # Write the results in hpatches eval format(0-10px mma, #features / #matches)
            results = np.hstack((mma,(pointsA.shape[0]+pointsB.shape[0])/2,matches.shape[0]))
            all_results = np.vstack((all_results,results))
            
            #Create subset and subsubset dicts to evaluate through subsets
            if not subset in subset_dict:
                subset_dict['all'][subset] = k
                subset_dict[subset] = {}
            
            if not subsubset in subset_dict[subset]:
                subset_dict[subset][subsubset] = k

            subset_dict[subset]['end'] = k    
        subset_dict['all']['end'] = k
           
        subset_list = sorted(subset_dict['all'].items(), key=lambda item: item[1])
        
        del subset_dict['all']
        
        subsubset_list = []
        for subset in subset_dict:
            subsubset_list.append(sorted(subset_dict[subset].items(), key=lambda item: item[1]))
        
        # subsubset_list = sorted([item for sublist in subsubset_list for item in sublist], key=lambda item: item[1])
        
    subset_results = {}
    for l in range(len(subset_list)-1):
        
        subset_results[subset_list[l][0]] = {subset_list[l][0]+'_all': [all_results[subset_list[l][1]:subset_list[l+1][1],i].mean() for i in range(10)] +
                                             [all_results[subset_list[l][1]:subset_list[l+1][1],10].mean().astype(int),
                                             all_results[subset_list[l][1]:subset_list[l+1][1],11].mean().astype(int)]}  
        
    subset_results['all'] = {'all': [all_results[subset_list[0][1]:subset_list[-1][1],i].mean() for i in range(10)] +
                                             [all_results[subset_list[0][1]:subset_list[-1][1],10].mean().astype(int),
                                             all_results[subset_list[0][1]:subset_list[-1][1],11].mean().astype(int)]} 
        
        
    if alg not in alg_results:
        alg_results[alg] = subset_results
                    
    #Write all_results for each algorithms as csv    
    np.savetxt(out_dir + '_mma' + ".csv", all_results, delimiter=",")
    
#Write overall_results for all algorithms and whole dataset as csv    
results_to_write = {}
for alg in alg_results:
    results_each_alg = []
    header1 = []
    header2 = []
    for subset in alg_results[alg]:
    
        for subsubset in alg_results[alg][subset]:
            results_each_alg.extend(alg_results[alg][subset][subsubset])
                
            header1.extend([" ", " ", " ", " ", " ", subsubset, " ", " ", " ", " ", " ", " "])
            header2.extend([f'{i},' for i in range(1,11)] + ["#Features", "#Matches"])
        
    results_to_write[alg] = results_each_alg
        
headers = [header1, header2]
tuples = list(zip(*headers))        
index = pd.MultiIndex.from_tuples(tuples, names=["Algorithms", " "])    

df = pd.DataFrame(np.vstack([np.array(results_to_write[a]) for a in results_to_write]), 
                  index=[alg_name for alg_name in results_to_write], columns=index)
df.to_csv(result_directory + '/' + dataset + '/' + 'overall_results_mma.csv',index=True, float_format='%.2f')


###################################################################################
###################################################################################
###################################################################################

#For calculating Homography Estimation

#Borrowed from https://github.com/GrumpyZhou/image-matching-toolbox hpatches_helper

from PIL import Image
import cv2

def eval_homography(p1s, p2s, h_gt, im1_path):
    # Estimate the homography between the matches using RANSAC
    try:
        H_pred, inliers = cv2.findHomography(p1s, p2s, cv2.USAC_MAGSAC, ransacReprojThreshold=3, maxIters=5000, confidence=0.9999)
    except:
        H_pred = None
        inliers = np.zeros(0)
    
    if H_pred is None:
        correctness = np.zeros(10)
    else:
        im = Image.open(im1_path)
        w, h = im.size
        corners = np.array([[0, 0, 1],
                            [0, w - 1, 1],
                            [h - 1, 0, 1],
                            [h - 1, w - 1, 1]])
        real_warped_corners = np.dot(corners, np.transpose(h_gt))
        real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
        warped_corners = np.dot(corners, np.transpose(H_pred))
        warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
        mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
        correctness = np.array([float(mean_dist <= i) for i in range (1,11)])
    return correctness, inliers

alg_results = {}
    
for i, alg in enumerate(algorithms):
    out_dir = result_directory + '/' + dataset + '/' + alg
    
    all_results = np.empty(shape=[0, 12])
    subset_dict = {}
    subset_dict['all'] = {}
    
    #For all pairs in image_pairs.txt do evaluation
    with open(dataset_dir + '/' + 'image_pairs.txt') as f:
        for k, line in enumerate(f):
            pairs = line.split(' ')
            subset = pairs[0].split('/')[0]
            subsubset = pairs[0].split('/')[1]
                           
            p1 = pairs[0].split('/')[2].split('.')[0]
            p2 = pairs[1].split('/')[2].split('.')[0]
            
        
            #Load output points and matches
            outputs = np.load(out_dir + '/' + 'outputs' + '/' + subset + '/' + subsubset + '/' + 
                              p1 + '_' + p2 + '_' + 'matches' + '.npz')
    
            pointsA = outputs['pointsA']
            pointsB = outputs['pointsB']
            matches = outputs['matches']
            
            #Load groundtruth homographies

            h_name = pairs[2].split('/')[2].split('\n')[0]
            h_gt = np.loadtxt(dataset_dir + '/' + subset + '/' + subsubset + '/' + h_name)
            im1_path = dataset_dir + '/' + subset + '/' + subsubset + '/' + p1 + '.ppm'

            pointsA_matched = pointsA[matches[:,0]]
            pointsB_matched = pointsB[matches[:,1]]
            
            hom_qual, inliers = eval_homography(pointsA_matched, pointsB_matched, h_gt, im1_path)
            
            if inliers.shape[0] > 0:
                number_of_inliers = sum(inliers > 0)[0]
            else:
                number_of_inliers = 0

            # Write the results in hpatches eval format(0-10px hom_qual, #matches / #inliers)
            results = np.hstack((hom_qual, matches.shape[0], number_of_inliers))
            all_results = np.vstack((all_results,results))
            
            #Create subset and subsubset dicts to evaluate through subsets
            if not subset in subset_dict:
                subset_dict['all'][subset] = k
                subset_dict[subset] = {}
            
            if not subsubset in subset_dict[subset]:
                subset_dict[subset][subsubset] = k

            subset_dict[subset]['end'] = k    
        subset_dict['all']['end'] = k
           
        subset_list = sorted(subset_dict['all'].items(), key=lambda item: item[1])
        
        del subset_dict['all']
        
        subsubset_list = []
        for subset in subset_dict:
            subsubset_list.append(sorted(subset_dict[subset].items(), key=lambda item: item[1]))
        
        # subsubset_list = sorted([item for sublist in subsubset_list for item in sublist], key=lambda item: item[1])
        
    subset_results = {}
    for l in range(len(subset_list)-1):
        
        subset_results[subset_list[l][0]] = {subset_list[l][0]+'_all': [all_results[subset_list[l][1]:subset_list[l+1][1],i].mean() for i in range(10)] +
                                             [all_results[subset_list[l][1]:subset_list[l+1][1],10].mean().astype(int),
                                             all_results[subset_list[l][1]:subset_list[l+1][1],11].mean().astype(int)]}  
        
    subset_results['all'] = {'all': [all_results[subset_list[0][1]:subset_list[-1][1],i].mean() for i in range(10)] +
                                             [all_results[subset_list[0][1]:subset_list[-1][1],10].mean().astype(int),
                                             all_results[subset_list[0][1]:subset_list[-1][1],11].mean().astype(int)]} 
        
        
    if alg not in alg_results:
        alg_results[alg] = subset_results
                    
    #Write all_results for each algorithms as csv    
    np.savetxt(out_dir + '_homography' + ".csv", all_results, delimiter=",")
    
#Write overall_results for all algorithms and whole dataset as csv    
results_to_write = {}
for alg in alg_results:
    results_each_alg = []
    header1 = []
    header2 = []
    for subset in alg_results[alg]:
    
        for subsubset in alg_results[alg][subset]:
            results_each_alg.extend(alg_results[alg][subset][subsubset])
                
            header1.extend([" ", " ", " ", " ", " ", subsubset, " ", " ", " ", " ", " ", " "])
            header2.extend([f'{i},' for i in range(1,11)] + ["#Matches", "#Inliers"])
        
    results_to_write[alg] = results_each_alg
        
headers = [header1, header2]
tuples = list(zip(*headers))        
index = pd.MultiIndex.from_tuples(tuples, names=["Algorithms", " "])    

df = pd.DataFrame(np.vstack([np.array(results_to_write[a]) for a in results_to_write]), 
                  index=[alg_name for alg_name in results_to_write], columns=index)
df.to_csv(result_directory + '/' + dataset + '/' + 'overall_results_homography.csv',index=True, float_format='%.2f')























                     
