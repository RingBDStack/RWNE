#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.0.0 
 
"""


import os
import sys






def main():
    log_name = sys.argv[1]
    counter = 0
    for line in open(log_name):
        line = line.strip()
        if "PID:" in line:
            PID = line.split("PID:")[1].strip()
            command = "kill {}".format(PID)
            print(command)
            os.system(command)
            counter +=1
    print("END! {} processes were killed!".format(counter))



if __name__ == '__main__':
    main()