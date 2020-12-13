#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.0.0 
 
"""

import numpy as np
import os

def get_classify_result(file, train_ratios=None):
    train_ratio_macro_micro = dict()
    macro_f1 = 0
    micro_f1 = 0
    train_ratio = '0'
    for line in open(file):
        line = line.strip()
        if line.startswith('train_ratio'):
            train_ratio_macro_micro[train_ratio] = [macro_f1, micro_f1]
            train_ratio = line.split('=')[-1].strip()
        elif line.startswith('Macro_F1'):
            macro_f1 = float(line.split('=')[-1].strip())
        elif line.startswith('Micro_F1'):
            micro_f1 = float(line.split('=')[-1].strip())
        else:
            continue
    train_ratio_macro_micro[train_ratio] = [macro_f1, micro_f1]

    if train_ratios==None:
        train_ratios = ["0."+str(v) for v in range(1,10)]

    macro_f1_list=[]
    micro_f1_list=[]
    for train_ratio in train_ratios:
        macro_f1_list.append(train_ratio_macro_micro[train_ratio][0])
        micro_f1_list.append(train_ratio_macro_micro[train_ratio][1])
    return macro_f1_list,micro_f1_list


file = "./classify.info"
train_ratios = ["0."+str(v) for v in range(1,10,2)]
macro_f1_list,micro_f1_list = get_classify_result(file, train_ratios)


print("Micro_F1:")
outstr = ""
for v in micro_f1_list:
    outstr+=" & %2.2f"%(np.floor(v*10000)/100)
outstr+=" \\\\"
print(outstr)

print("Macro_F1:")
outstr = ""
for v in macro_f1_list:
    outstr += " & %2.2f" % (np.floor(v * 10000) / 100)
outstr+=" \\\\"
print(outstr)


