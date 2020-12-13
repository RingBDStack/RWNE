#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.0.0

reference:
    https://www.cnblogs.com/hei-hei-hei/p/7216434.html
    https://blog.csdn.net/u011961856/article/details/77884946
"""


import os
import numpy as np
import time

MAX_GPU = 10000 # for GPU171

while True:
    # watch the GPU memory, and save the info into the file 'watchdog.output'
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > watchdog.output')
    # get the GPU memory.
    memory_gpu=[int(x.strip().split()[2].strip()) for x in open('watchdog.output','r').readlines()]

    # get the gpu device ID who has the maximum memory.
    deviceID = np.argmax(memory_gpu)
    max_gpu_mem = memory_gpu[deviceID]
    # os.environ['CUDA_VISIBLE_DEVICES']=str(deviceID)

    # the lower bound for program running (MB)
    lower_bound = 6000

    if max_gpu_mem >= lower_bound:
        if max_gpu_mem >= MAX_GPU:
            gmf = 0.17
        else:
            gmf = (max_gpu_mem/MAX_GPU)/5.0
        print("GPU/{} : {} MiB".format(deviceID, max_gpu_mem))
        os.system('nohup bash lp_line_single_youtube.sh 0.7 1 5 1 1 %d %.5f > mylog_single.out 2>&1 & '%(deviceID, gmf))
        break

    os.system('rm watchdog.output')
    time.sleep(1)
print("process-1 started...\n")
os.system('rm watchdog.output')
time.sleep(1000)


while True:
    # watch the GPU memory, and save the info into the file 'watchdog.output'
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > watchdog.output')
    # get the GPU memory.
    memory_gpu=[int(x.strip().split()[2].strip()) for x in open('watchdog.output','r').readlines()]

    # get the gpu device ID who has the maximum memory.
    deviceID = np.argmax(memory_gpu)
    max_gpu_mem = memory_gpu[deviceID]
    # os.environ['CUDA_VISIBLE_DEVICES']=str(deviceID)

    # the lower bound for program running (MB)
    lower_bound = 6000

    if max_gpu_mem >= lower_bound:
        if max_gpu_mem >= MAX_GPU:
            gmf = 0.17
        else:
            gmf = (max_gpu_mem/MAX_GPU)/5.0
        print("GPU/{} : {} MiB\n".format(deviceID, max_gpu_mem))
        os.system('nohup bash lp_line_single_youtube.sh 0.7 6 10 1 1 %d %.5f > mylog_single.out 2>&1 & '%(deviceID, gmf))
        break

    os.system('rm watchdog.output')
    time.sleep(1)
print("process-2 started...")
os.system('rm watchdog.output')
time.sleep(1000)




while True:
    # watch the GPU memory, and save the info into the file 'watchdog.output'
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > watchdog.output')
    # get the GPU memory.
    memory_gpu=[int(x.strip().split()[2].strip()) for x in open('watchdog.output','r').readlines()]

    # get the gpu device ID who has the maximum memory.
    deviceID = np.argmax(memory_gpu)
    max_gpu_mem = memory_gpu[deviceID]
    # os.environ['CUDA_VISIBLE_DEVICES']=str(deviceID)

    # the lower bound for program running (MB)
    lower_bound = 6000

    if max_gpu_mem >= lower_bound:
        if max_gpu_mem >= MAX_GPU:
            gmf = 0.17
        else:
            gmf = (max_gpu_mem/MAX_GPU)/5.0
        print("GPU/{} : {} MiB\n".format(deviceID, max_gpu_mem))
        os.system('nohup bash lp_line_single_youtube.sh 0.6 1 5 1 1 %d %.5f > mylog_single.out 2>&1 & '%(deviceID, gmf))
        break

    os.system('rm watchdog.output')
    time.sleep(1)
print("process-3 started...")
os.system('rm watchdog.output')
time.sleep(1000)


while True:
    # watch the GPU memory, and save the info into the file 'watchdog.output'
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > watchdog.output')
    # get the GPU memory.
    memory_gpu=[int(x.strip().split()[2].strip()) for x in open('watchdog.output','r').readlines()]

    # get the gpu device ID who has the maximum memory.
    deviceID = np.argmax(memory_gpu)
    max_gpu_mem = memory_gpu[deviceID]
    # os.environ['CUDA_VISIBLE_DEVICES']=str(deviceID)

    # the lower bound for program running (MB)
    lower_bound = 6000

    if max_gpu_mem >= lower_bound:
        if max_gpu_mem >= MAX_GPU:
            gmf = 0.17
        else:
            gmf = (max_gpu_mem/MAX_GPU)/5.0
        print("GPU/{} : {} MiB\n".format(deviceID, max_gpu_mem))
        os.system('nohup bash lp_line_single_youtube.sh 0.6 6 10 1 1 %d %.5f > mylog_single.out 2>&1 & '%(deviceID, gmf))
        break

    os.system('rm watchdog.output')
    time.sleep(1)
print("process-4 started...")
os.system('rm watchdog.output')
time.sleep(1000)




