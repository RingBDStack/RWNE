#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.0.0 
 
"""

import os
import shutil

path= "./"
file_tags = ["RUN_LOG"]
tags = ["EVAL_PID"]
for file in os.listdir(path):
    for file_tag in file_tags:
        if file_tag in file:
            file_path = os.path.join(path, file)
            for line in open(file_path, 'r'):
                line = line.strip()
                for tag in tags:
                    if line.startswith(tag):
                        print(line)
            os.remove(file_path)



