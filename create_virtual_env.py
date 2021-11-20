#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:15:00 2021

@authors: kutalmisince and ufukefe
"""

import os

def create_virtual_env(name, directory):
    
    env_list = os.popen('conda env list').read()
    exist = name in env_list
    
    if not exist:       
        os.system('conda env create -f' + directory + '/' + 'environment.yml')
        
        return 1
    
    else:
        print('The environment is already exists')
        return 0

