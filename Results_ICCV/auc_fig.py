#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 01:48:19 2021

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
    
    
    
### AUC Dictionaries
    
mma_auc = {}
homography_auc = {}

ratio_thr = 0.1

for pixel_thr in range(1,11):
    mma_auc[str(ratio_thr)] = {}
    for key in mma_results_all[str(pixel_thr)].keys():
        mma_auc[str(ratio_thr)][key] = []
    ratio_thr = round(ratio_thr + 0.1,1)

for key in mma_results_all[str(pixel_thr)].keys():
    for k, ratio_thr in enumerate(np.round(np.arange(0.1, 1.1, 0.1),1)):
        for pixel_thr in range(1,11):
            mma_auc[str(ratio_thr)][key].append(mma_results_all[str(pixel_thr)][key][k])
    
    
ratio_thr = 0.1

for pixel_thr in range(1,11):
    homography_auc[str(ratio_thr)] = {}
    for key in homography_results_all[str(pixel_thr)].keys():
        homography_auc[str(ratio_thr)][key] = []  
    ratio_thr = round(ratio_thr + 0.1,1)

for key in homography_results_all[str(pixel_thr)].keys():
    for k, ratio_thr in enumerate(np.round(np.arange(0.1, 1.1, 0.1),1)):
        for pixel_thr in range(1,11):
            homography_auc[str(ratio_thr)][key].append(homography_results_all[str(pixel_thr)][key][k])
       

### Calculate AUCs

mma_auc_values = {}
homography_auc_values = {}


for ratio_thr in np.round(np.arange(0.1, 1.1, 0.1),1):
    for key in mma_auc[str(ratio_thr)].keys():
        if key not in mma_auc_values:
            mma_auc_values[key] = []
        mma_auc_values[key].append(np.round(np.trapz(np.asarray(mma_auc[str(ratio_thr)][key]),dx=1),2))

for ratio_thr in np.round(np.arange(0.1, 1.1, 0.1),1):
    for key in homography_auc[str(ratio_thr)].keys():
        if key not in homography_auc_values:
            homography_auc_values[key] = []
        homography_auc_values[key].append(np.round(np.trapz(np.asarray(homography_auc[str(ratio_thr)][key]),dx=1),2))


mma_best_auc_values = {}
homography_best_auc_values = {}

# multiply by 100/9 to get auc percentage since the max are is 9 (1-10 pixels)
for key in mma_auc_values.keys():
    mma_best_auc_values[key] = [round(max(mma_auc_values[key])*100/9,1), 
                                round(0.1*(mma_auc_values[key].index(max(mma_auc_values[key]))+1),1)]

for key in homography_auc_values.keys():
    homography_best_auc_values[key] = [round(max(homography_auc_values[key])*100/9,1), 
                                round(0.1*(homography_auc_values[key].index(max(homography_auc_values[key]))+1),1)]

mma_best_auc_values_pd = pd.DataFrame (mma_best_auc_values).T
mma_best_auc_values_pd.to_csv("mma_best_auc_values.csv")

homography_best_auc_values_pd = pd.DataFrame (homography_best_auc_values).T
homography_best_auc_values_pd.to_csv("homography_best_auc_values.csv")


mma_best_auc = {}
homography_best_auc = {}

for key in mma_best_auc_values.keys():
    mma_best_auc[key] = mma_auc[str(mma_best_auc_values[key][1])][key]

for key in homography_best_auc_values.keys():
    homography_best_auc[key] = homography_auc[str(homography_best_auc_values[key][1])][key]
    

### DRAW RESULTS

#Generate random colors
colors = ['red', 'orange', 'rosybrown', 'purple' ,'magenta', 'steelblue', 'deepskyblue', 'olive', 'limegreen']
pixel_thresholds = np.arange(1,11)
fig,axes=plt.subplots(nrows=2, ncols=2, figsize=(14, 14))

for i, key in enumerate(mma_best_auc):
    
    if key in ('sift', 'surf', 'orb', 'kaze', 'akaze'):
        axes[0][0].plot(pixel_thresholds, mma_best_auc[key], color = colors[i], Linestyle='dashed', LineWidth = 2.8, label=key)
        # axes[0][0].plot(pixel_thresholds, homography_auc[str(mma_best_auc_values[key][1])][key], color = colors[i], marker = 'o', Linestyle='None', markersize = 7)

    elif key in ('superpoint', 'superglue', 'patch2pix', 'dfm'):
        axes[0][0].plot(pixel_thresholds, mma_best_auc[key], color = colors[i], LineWidth = 2.8, label=key)
        # axes[0][0].plot(pixel_thresholds, homography_auc[str(mma_best_auc_values[key][1])][key], color = colors[i], marker = 'o', Linestyle='None', markersize = 7)      

axes[0][0].set_xlim(1,10)
axes[0][0].set_ylim(0,1)
axes[0][0].set_xticks(pixel_thresholds)
axes[0][0].set_yticks(np.arange(0, 1.1, 0.1))

axes[0][0].set_title('MMA curves giving the best AUC', fontsize=17, fontweight = 'bold', pad = 17)
axes[0][0].grid()
axes[0][0].set_ylabel('Accuracy', fontsize=17)
axes[0][0].set_xlabel('Pixel Threshold', fontsize=17) 
leg = axes[0][0].legend(loc="lower right", labelspacing = 0.25, handlelength=2, handletextpad=1, fancybox=True)
for n, text in enumerate( leg.texts ): text.set_color( colors[n] )   


for i, key in enumerate(homography_best_auc):
    if key in ('sift', 'surf', 'orb', 'kaze', 'akaze'):
        # axes[0][1].plot(pixel_thresholds, mma_auc[str(homography_best_auc_values[key][1])][key], color = colors[i], Linestyle='dashed', LineWidth = 2.8, label=key)
        axes[0][1].plot(pixel_thresholds, homography_best_auc[key], color = colors[i], marker = '*', Linestyle='None', markersize = 7, label=key)
 
    elif key in ('superpoint', 'superglue', 'patch2pix', 'dfm'):
        # axes[0][1].plot(pixel_thresholds, mma_auc[str(homography_best_auc_values[key][1])][key], color = colors[i], LineWidth = 2.8, label=key)
        axes[0][1].plot(pixel_thresholds, homography_best_auc[key], color = colors[i], marker = 'o', Linestyle='None', markersize = 7, label=key)

    
axes[0][1].set_xlim(1,10)
axes[0][1].set_ylim(0,1)
axes[0][1].set_xticks(pixel_thresholds)
axes[0][1].set_yticks(np.arange(0, 1.1, 0.1))

axes[0][1].set_title('HEA curves giving the best AUC', fontsize=17, fontweight = 'bold', pad = 17)
axes[0][1].grid()
axes[0][1].set_ylabel('Accuracy', fontsize=17)
axes[0][1].set_xlabel('Pixel Threshold', fontsize=17)     
leg = axes[0][1].legend(loc="lower right", labelspacing = 0.25, handlelength=2, handletextpad=1, fancybox=True)
for n, text in enumerate( leg.texts ): text.set_color( colors[n] )   




for i, key in enumerate(mma_best_auc):
    
    if key in ('sift', 'surf', 'orb', 'kaze', 'akaze'):
        # axes[1][0].plot(pixel_thresholds, mma_best_auc[key], color = colors[i], Linestyle='dashed', LineWidth = 2.8, label=key)
        axes[1][0].plot(pixel_thresholds, homography_auc[str(mma_best_auc_values[key][1])][key], color = colors[i], marker = '*', Linestyle='None', markersize = 7, label=key)

    elif key in ('superpoint', 'superglue', 'patch2pix', 'dfm'):
        # axes[1][0].plot(pixel_thresholds, mma_best_auc[key], color = colors[i], LineWidth = 2.8, label=key)
        axes[1][0].plot(pixel_thresholds, homography_auc[str(mma_best_auc_values[key][1])][key], color = colors[i], marker = 'o', Linestyle='None', markersize = 7, label=key)      

axes[1][0].set_xlim(1,10)
axes[1][0].set_ylim(0,1)
axes[1][0].set_xticks(pixel_thresholds)
axes[1][0].set_yticks(np.arange(0, 1.1, 0.1))

axes[1][0].set_title('Relevant HEA Results', fontsize=17, fontweight = 'bold', pad = 17)
axes[1][0].grid()
axes[1][0].set_ylabel('Accuracy', fontsize=17)
axes[1][0].set_xlabel('Pixel Threshold', fontsize=17) 


for i, key in enumerate(homography_best_auc):
    if key in ('sift', 'surf', 'orb', 'kaze', 'akaze'):
        axes[1][1].plot(pixel_thresholds, mma_auc[str(homography_best_auc_values[key][1])][key], color = colors[i], Linestyle='dashed', LineWidth = 2.8, label=key)
        # axes[1][1].plot(pixel_thresholds, homography_best_auc[key], color = colors[i], marker = '*', Linestyle='None', markersize = 7, label=key)
 
    elif key in ('superpoint', 'superglue', 'patch2pix', 'dfm'):
        axes[1][1].plot(pixel_thresholds, mma_auc[str(homography_best_auc_values[key][1])][key], color = colors[i], LineWidth = 2.8, label=key)
        # axes[1][1].plot(pixel_thresholds, homography_best_auc[key], color = colors[i], marker = 'o', Linestyle='None', markersize = 7, label=key)

    
axes[1][1].set_xlim(1,10)
axes[1][1].set_ylim(0,1)
axes[1][1].set_xticks(pixel_thresholds)
axes[1][1].set_yticks(np.arange(0, 1.1, 0.1))

axes[1][1].set_title('Relevant MMA Results', fontsize=17, fontweight = 'bold', pad = 17)
axes[1][1].grid()
axes[1][1].set_ylabel('Accuracy', fontsize=17)
axes[1][1].set_xlabel('Pixel Threshold', fontsize=17)     


plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.7, 
                    top=0.7, 
                    wspace=0.4, 
                    hspace=0.4)

plt.savefig('./' + 'best_auc.eps', format='eps', bbox_inches='tight', pad_inches=0)
#plt.savefig('./' + 'best_auc.png', format='png')
plt.show()   
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    













