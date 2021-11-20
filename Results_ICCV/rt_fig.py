#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 04:37:58 2021

@author: ufuk
"""

import numpy as np
import collections
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd


###                             ###
###    CLASSICAL ALGORITHMS     ###
###                             ###

directory = 'hpatches_classical'

###       MMA EXPERIMENTS       ###

#Construct rr (ratio results) dictionaries
rri = collections.defaultdict(list)
rrv = collections.defaultdict(list)
rra = collections.defaultdict(list)

with open(directory + '/' + os.listdir(directory)[0] + '/' + 'overall_results_mma.csv') as f:  
    for k, line in enumerate(f):  
         line = line.split(',')
         if k>=2:
             rri[line[0]] = []
             rrv[line[0]] = []
             rra[line[0]] = []

mma_results_illum_classical = {}
mma_results_vp_classical = {}
mma_results_all_classical = {}

for pixel_threshold in range(1,11):
    for ratio_threshold in np.arange(0.1, 1.1, 0.1):
        with open(directory + '/' + str(round(ratio_threshold, 2)) + '/' + 'overall_results_mma.csv') as f:  
            for k, line in enumerate(f):  
                 line = line.split(',')
                 if k>=2:
                     rri[line[0]].append(float(line[pixel_threshold]))
                     rrv[line[0]].append(float(line[pixel_threshold+12])) 
                     rra[line[0]].append(float(line[pixel_threshold+24]))  
                     
                     
    mma_results_illum_classical[str(pixel_threshold)] = dict(rri)
    mma_results_vp_classical[str(pixel_threshold)] = dict(rrv)
    mma_results_all_classical[str(pixel_threshold)] = dict(rra)
    for key in rri.keys(): rri[key] = []
    for key in rrv.keys(): rrv[key] = []
    for key in rra.keys(): rra[key] = []
    
    
###       HEA EXPERIMENTS       ###

#Construct rr (ratio results) dictionaries
hri = collections.defaultdict(list)
hrv = collections.defaultdict(list)
hra = collections.defaultdict(list)

with open(directory + '/' + os.listdir(directory)[0] + '/' + 'overall_results_homography.csv') as f:  
    for k, line in enumerate(f):  
         line = line.split(',')
         if k>=2:
             hri[line[0]] = []
             hrv[line[0]] = []
             hra[line[0]] = []

homography_results_illum_classical = {}
homography_results_vp_classical = {}
homography_results_all_classical = {}


for pixel_threshold in range(1,11):
    for ratio_threshold in np.arange(0.1, 1.1, 0.1):
        with open(directory + '/' + str(round(ratio_threshold, 2)) + '/' + 'overall_results_homography.csv') as f:  
            for k, line in enumerate(f):  
                 line = line.split(',')
                 if k>=2:
                     hri[line[0]].append(float(line[pixel_threshold]))
                     hrv[line[0]].append(float(line[pixel_threshold+12])) 
                     hra[line[0]].append(float(line[pixel_threshold+24]))  
                     
                     
    homography_results_illum_classical[str(pixel_threshold)] = dict(hri)
    homography_results_vp_classical[str(pixel_threshold)] = dict(hrv)
    homography_results_all_classical[str(pixel_threshold)] = dict(hra)
    for key in hri.keys(): hri[key] = []
    for key in hrv.keys(): hrv[key] = []
    for key in hra.keys(): hra[key] = []
    

###                             ###
###  LEARNING-BASED ALGORITHMS  ###
###                             ###   
    
directory = 'hpatches_deep'

###       MMA EXPERIMENTS       ###

#Construct rr (ratio results) dictionaries
rri = collections.defaultdict(list)
rrv = collections.defaultdict(list)
rra = collections.defaultdict(list)

with open(directory + '/' + os.listdir(directory)[0] + '/' + 'overall_results_mma.csv') as f:  
    for k, line in enumerate(f):  
         line = line.split(',')
         if k>=2:
             rri[line[0]] = []
             rrv[line[0]] = []
             rra[line[0]] = []

mma_results_illum_deep = {}
mma_results_vp_deep = {}
mma_results_all_deep = {}

for pixel_threshold in range(1,11):
    for ratio_threshold in np.arange(0.0, 1.1, 0.1):
            with open(directory + '/' + str(round(ratio_threshold, 2)) + '/' + 'overall_results_mma.csv') as f:  
                for k, line in enumerate(f):  
                     line = line.split(',')
                     if k>=2:
                         rri[line[0]].append(float(line[pixel_threshold]))
                         rrv[line[0]].append(float(line[pixel_threshold+12])) 
                         rra[line[0]].append(float(line[pixel_threshold+24]))  
                     
                          
    mma_results_illum_deep[str(pixel_threshold)] = dict(rri)
    mma_results_vp_deep[str(pixel_threshold)] = dict(rrv)
    mma_results_all_deep[str(pixel_threshold)] = dict(rra)
    for key in rri.keys(): rri[key] = []
    for key in rrv.keys(): rrv[key] = []
    for key in rra.keys(): rra[key] = []
    
###       HEA EXPERIMENTS       ###

#Construct rr (ratio results) dictionaries
hri = collections.defaultdict(list)
hrv = collections.defaultdict(list)
hra = collections.defaultdict(list)

with open(directory + '/' + os.listdir(directory)[0] + '/' + 'overall_results_homography.csv') as f:  
    for k, line in enumerate(f):  
         line = line.split(',')
         if k>=2:
             hri[line[0]] = []
             hrv[line[0]] = []
             hra[line[0]] = []

homography_results_illum_deep = {}
homography_results_vp_deep = {}
homography_results_all_deep = {}


for pixel_threshold in range(1,11):
    for ratio_threshold in np.arange(0.0, 1.1, 0.1):
        with open(directory + '/' + str(round(ratio_threshold, 2)) + '/' + 'overall_results_homography.csv') as f:  
            for k, line in enumerate(f):  
                 line = line.split(',')
                 if k>=2:
                     hri[line[0]].append(float(line[pixel_threshold]))
                     hrv[line[0]].append(float(line[pixel_threshold+12])) 
                     hra[line[0]].append(float(line[pixel_threshold+24]))  
                     
                     
    homography_results_illum_deep[str(pixel_threshold)] = dict(hri)
    homography_results_vp_deep[str(pixel_threshold)] = dict(hrv)
    homography_results_all_deep[str(pixel_threshold)] = dict(hra)
    for key in hri.keys(): hri[key] = []
    for key in hrv.keys(): hrv[key] = []
    for key in hra.keys(): hra[key] = []
        

#SuperGlue and Patch2pix looks for confidence, so reverse them and keep first 10
for key in mma_results_all_deep.keys():
    mma_results_all_deep[key]['superpoint'] = mma_results_all_deep[key]['superpoint'][1:11]
    mma_results_all_deep[key]['dfm'] = mma_results_all_deep[key]['dfm'][1:11]
    mma_results_all_deep[key]['superglue'] = mma_results_all_deep[key]['superglue'][::-1][1:11]
    mma_results_all_deep[key]['patch2pix'] = mma_results_all_deep[key]['patch2pix'][::-1][1:11]
    
    mma_results_illum_deep[key]['superpoint'] = mma_results_illum_deep[key]['superpoint'][1:11]
    mma_results_illum_deep[key]['dfm'] = mma_results_illum_deep[key]['dfm'][1:11]
    mma_results_illum_deep[key]['superglue'] = mma_results_illum_deep[key]['superglue'][::-1][1:11]
    mma_results_illum_deep[key]['patch2pix'] = mma_results_illum_deep[key]['patch2pix'][::-1][1:11]
    
    mma_results_vp_deep[key]['superpoint'] = mma_results_vp_deep[key]['superpoint'][1:11]
    mma_results_vp_deep[key]['dfm'] = mma_results_vp_deep[key]['dfm'][1:11]
    mma_results_vp_deep[key]['superglue'] = mma_results_vp_deep[key]['superglue'][::-1][1:11]
    mma_results_vp_deep[key]['patch2pix'] = mma_results_vp_deep[key]['patch2pix'][::-1][1:11]

for key in homography_results_all_deep.keys():
    homography_results_all_deep[key]['superpoint'] = homography_results_all_deep[key]['superpoint'][1:11]
    homography_results_all_deep[key]['dfm'] = homography_results_all_deep[key]['dfm'][1:11]
    homography_results_all_deep[key]['superglue'] = homography_results_all_deep[key]['superglue'][::-1][1:11]
    homography_results_all_deep[key]['patch2pix'] = homography_results_all_deep[key]['patch2pix'][::-1][1:11]
    
    homography_results_illum_deep[key]['superpoint'] = homography_results_illum_deep[key]['superpoint'][1:11]
    homography_results_illum_deep[key]['dfm'] = homography_results_illum_deep[key]['dfm'][1:11]
    homography_results_illum_deep[key]['superglue'] = homography_results_illum_deep[key]['superglue'][::-1][1:11]
    homography_results_illum_deep[key]['patch2pix'] = homography_results_illum_deep[key]['patch2pix'][::-1][1:11]
    
    homography_results_vp_deep[key]['superpoint'] = homography_results_vp_deep[key]['superpoint'][1:11]
    homography_results_vp_deep[key]['dfm'] = homography_results_vp_deep[key]['dfm'][1:11]
    homography_results_vp_deep[key]['superglue'] = homography_results_vp_deep[key]['superglue'][::-1][1:11]
    homography_results_vp_deep[key]['patch2pix'] = homography_results_vp_deep[key]['patch2pix'][::-1][1:11]   
    

### MERGE ALL RESULTS

mma_results_illum = {}
mma_results_vp = {}
mma_results_all = {}

homography_results_illum = {}
homography_results_vp = {}
homography_results_all = {}


for key in mma_results_all_deep.keys():
    mma_results_all[key] = mma_results_all_classical[key].copy()
    mma_results_all[key].update(mma_results_all_deep[key])
    
    mma_results_illum[key] = mma_results_illum_classical[key].copy()
    mma_results_illum[key].update(mma_results_illum_deep[key])
    
    mma_results_vp[key] = mma_results_vp_classical[key].copy()
    mma_results_vp[key].update(mma_results_vp_deep[key])
    
    
for key in homography_results_all_deep.keys():
    homography_results_all[key] = homography_results_all_classical[key].copy()
    homography_results_all[key].update(homography_results_all_deep[key])
    
    homography_results_illum[key] = homography_results_illum_classical[key].copy()
    homography_results_illum[key].update(homography_results_illum_deep[key])
    
    homography_results_vp[key] = homography_results_vp_classical[key].copy()
    homography_results_vp[key].update(homography_results_vp_deep[key])
    
    
### DRAW RESULTS

#Generate random colors
colors = ['red', 'orange', 'rosybrown', 'purple' ,'magenta', 'steelblue', 'deepskyblue', 'olive', 'limegreen']
pixel_thresholds = [1, 3, 5, 10]
fig,axes=plt.subplots(nrows=len(pixel_thresholds),ncols=3, figsize=(21, len(pixel_thresholds)*7))



### For over_all
for k, pixel_thr in enumerate(pixel_thresholds):

    for i, key in enumerate(mma_results_all['1']):
        
        if key in ('sift', 'surf', 'orb', 'kaze', 'akaze'):
            axes[k][0].plot(np.arange(0.1, 1.1, 0.1), mma_results_all[str(pixel_thr)][key], color = colors[i], Linestyle='dashed', LineWidth = 2.8, label=key)
            axes[k][0].plot(np.arange(0.1, 1.1, 0.1), homography_results_all[str(pixel_thr)][key], color = colors[i], marker = 'o', Linestyle='None', markersize = 7)

        elif key in ('superpoint', 'superglue', 'patch2pix', 'dfm'):
            axes[k][0].plot(np.arange(0.1, 1.1, 0.1), mma_results_all[str(pixel_thr)][key], color = colors[i], LineWidth = 2.8, label=key)
            axes[k][0].plot(np.arange(0.1, 1.1, 0.1), homography_results_all[str(pixel_thr)][key], color = colors[i], marker = 'o', Linestyle='None', markersize = 7) 
                   
    axes[k][0].set_xlim(0,1)
    axes[k][0].set_ylim(0,1)
    axes[k][0].set_xticks(np.arange(0, 1.1, 0.1))
    axes[k][0].set_yticks(np.arange(0, 1.1, 0.1))
    if k == 0:
       axes[k][0].set_title('Overall', fontsize=21, fontweight = 'bold', pad = 17)
    axes[k][0].grid()
    if pixel_thr == 1:
        axes[k][0].set_ylabel('Accuracy ' + '(≤ ' + f'{pixel_thr}' + '-pixel' + ')', fontsize=14)
    else:
        axes[k][0].set_ylabel('Accuracy ' + '(≤ ' + f'{pixel_thr}' + '-pixels' + ')', fontsize=14)
    if k == len(pixel_thresholds)-1:
        axes[k][0].set_xlabel('Threshold', fontsize=17) 
    
    for i, key in enumerate(mma_results_all['1']):
        
        if key in ('sift', 'surf', 'orb', 'kaze', 'akaze'):
            axes[k][1].plot(np.arange(0.1, 1.1, 0.1), mma_results_illum[str(pixel_thr)][key], color = colors[i], Linestyle='dashed', LineWidth = 2.8, label=key)
            axes[k][1].plot(np.arange(0.1, 1.1, 0.1), homography_results_illum[str(pixel_thr)][key], color = colors[i], marker = 'o', Linestyle='None', markersize = 7)

        elif key in ('superpoint', 'superglue', 'patch2pix', 'dfm'):
            axes[k][1].plot(np.arange(0.1, 1.1, 0.1), mma_results_illum[str(pixel_thr)][key], color = colors[i], LineWidth = 2.8, label=key)
            axes[k][1].plot(np.arange(0.1, 1.1, 0.1), homography_results_illum[str(pixel_thr)][key], color = colors[i], marker = 'o', Linestyle='None', markersize = 7) 
                   
    axes[k][1].set_xlim(0,1)
    axes[k][1].set_ylim(0,1)
    axes[k][1].set_xticks(np.arange(0, 1.1, 0.1))
    axes[k][1].set_yticks(np.arange(0, 1.1, 0.1))
    if k == 0:
        axes[k][1].set_title('Illumination', fontsize=21, fontweight = 'bold', pad = 17)
    axes[k][1].grid()
    if k == len(pixel_thresholds)-1:
       axes[k][1].set_xlabel('Threshold', fontsize=17) 
    
    
    for i, key in enumerate(mma_results_all['1']):
        
        if key in ('sift', 'surf', 'orb', 'kaze', 'akaze'):
            axes[k][2].plot(np.arange(0.1, 1.1, 0.1), mma_results_vp[str(pixel_thr)][key], color = colors[i], Linestyle='dashed', LineWidth = 2.8, label=key)
            axes[k][2].plot(np.arange(0.1, 1.1, 0.1), homography_results_vp[str(pixel_thr)][key], color = colors[i], marker = 'o', Linestyle='None', markersize = 7)

        elif key in ('superpoint', 'superglue', 'patch2pix', 'dfm'):
            axes[k][2].plot(np.arange(0.1, 1.1, 0.1), mma_results_vp[str(pixel_thr)][key], color = colors[i], LineWidth = 2.8, label=key)
            axes[k][2].plot(np.arange(0.1, 1.1, 0.1), homography_results_vp[str(pixel_thr)][key], color = colors[i], marker = 'o', Linestyle='None', markersize = 7) 
                   
    axes[k][2].set_xlim(0,1)
    axes[k][2].set_ylim(0,1)
    axes[k][2].set_xticks(np.arange(0, 1.1, 0.1))
    axes[k][2].set_yticks(np.arange(0, 1.1, 0.1))
    if k == 0:
        axes[k][2].set_title('Viewpoint', fontsize=21, fontweight = 'bold', pad = 17)
    axes[k][2].grid()
    if k == len(pixel_thresholds)-1:
        axes[k][2].set_xlabel('Threshold', fontsize=17) 
    
    black_line = mlines.Line2D([], [], color='black', label='Mean Matching Accuracy (MMA)')
    black_o = mlines.Line2D([], [], color='black', marker='o', markersize=5, 
                                  Linestyle='None', label='Homography Estimation Accuracy')
    accuracy_legend  = plt.legend(handles=[black_line, black_o], loc="lower left", labelspacing = 0.25, prop={'size': 11}, handlelength=1)
    
    leg = plt.legend(loc="lower right", labelspacing = 0.15, handlelength=0, handletextpad=0, fancybox=True, prop={'size': 14})
    for n, text in enumerate( leg.texts ): text.set_color( colors[n] )
      
    plt.gca().add_artist(accuracy_legend)
    
plt.savefig('./' + '21by' + f'{len(pixel_thresholds)*7}.eps', format='eps', bbox_inches='tight', pad_inches=0)
#plt.savefig('./' + '21by' + f'{len(pixel_thresholds)*7}.png', format='png')
plt.show()
    

### Find the threshold giving the best mma and hea

mma_best = {}
homography_best = {}    
    
for pixel_thr in pixel_thresholds:
    mma_best[str(pixel_thr)] = {}
    for key in mma_results_all[str(pixel_thr)].keys():
        mma_best[str(pixel_thr)][key] = [max(mma_results_all[str(pixel_thr)][key]), 
                                         round(0.1*(mma_results_all[str(pixel_thr)][key].index(max(mma_results_all[str(pixel_thr)][key]))+1),1)]

for pixel_thr in pixel_thresholds:
    homography_best[str(pixel_thr)] = {}
    for key in homography_results_all[str(pixel_thr)].keys():
        homography_best[str(pixel_thr)][key] = [max(homography_results_all[str(pixel_thr)][key]), 
                                         round(0.1*(homography_results_all[str(pixel_thr)][key].index(max(homography_results_all[str(pixel_thr)][key]))+1),1)]
        
    

mma_best_pd = pd.DataFrame (mma_best)
mma_best_pd.to_csv("mma_best.csv")

homography_best_pd = pd.DataFrame (homography_best)
homography_best_pd.to_csv("homography_best.csv")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
