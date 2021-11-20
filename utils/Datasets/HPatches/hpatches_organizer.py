#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:15:00 2021

@authors: kutalmisince and ufukefe
"""

import os
import shutil

original_name = 'hpatches-sequences-release'
new_name = 'hpatches'
subsubset_list = sorted(os.listdir(original_name))

if not os.path.exists(new_name):
    os.makedirs(new_name)

with open (new_name + '/' 'image_pairs.txt' , 'a') as f:
    for subsubset in subsubset_list:
        
        if subsubset[0] == 'i':
            subset = 'illumination'
            for i in range(2,7):
                pair = ['1.ppm', f'{i}.ppm', f'H_1_{i}']
                for ind in pair:
                    if not os.path.exists(new_name + '/' + subset + '/' + subsubset):
                        os.makedirs(new_name + '/' + subset + '/' + subsubset)
                    shutil.copy(original_name + '/' + subsubset + '/' + ind,
                                new_name + '/' + subset + '/' + subsubset + '/' +  subsubset + '_' + ind)
                    pair_to_write = [subset + '/' + subsubset + '/' +  subsubset + '_' + ind for ind in pair]
                f.write(pair_to_write[0] + ' ' + pair_to_write[1] + ' ' + pair_to_write[2] + '\n')
                    
                    
        elif subsubset[0] == 'v':
            subset = 'viewpoint'
            for i in range(2,7):
                pair = ['1.ppm', f'{i}.ppm', f'H_1_{i}']
                for ind in pair:
                    if not os.path.exists(new_name + '/' + subset + '/' + subsubset):
                        os.makedirs(new_name + '/' + subset + '/' + subsubset)
                    shutil.copy(original_name + '/' + subsubset + '/' + ind,
                                new_name + '/' + subset + '/' + subsubset + '/' +  subsubset + '_' + ind)
                    pair_to_write = [subset + '/' + subsubset + '/' +  subsubset + '_' + ind for ind in pair]
                f.write(pair_to_write[0] + ' ' + pair_to_write[1] + ' ' + pair_to_write[2] + '\n')
                        
