#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.0.0 
 
"""

import os
import sys
import logging
import time
import shutil


dataset_name_list=("blogcatalog", "flickr", "youtube", "cora", "citeseer", "pubmed", "dblp", "wiki", "PPI", "emailEu")
dataset_labels_list=(39, 195, 47, 7, 6, 3, 4, 17, 50, 42)

data_dir="/home/LAB/heyu/NRL/data/"
source_data_dir="/home/LAB/heyu/NRL/DataSets_bak/"


def check_labels():
    for id, dataset_name in enumerate(dataset_name_list):
        print("\n"+dataset_name+"::::::::")
        dataset_labels = dataset_labels_list[id]
        label_file = os.path.join(data_dir, dataset_name, dataset_name+".labels")
        if not os.path.exists(label_file):
            print(label_file+" not exists.")
            return
        total_labels = set()
        for line in open(label_file):
            linelist = line.strip().split('\t')
            if len(linelist) == 0:
                print('empty line!')
                continue
            elif len(linelist) == 1:
                print('label may be lacked in line: %s' % line)
                continue
            for label_id in [int(v) for v in linelist[1:]]:
                total_labels.add(label_id)
        print("expected_labels_size={},,,real_labels_size={}".format(dataset_labels, len(total_labels)))
        for i in range(dataset_labels):
            if i not in total_labels:
                print("!!!!!!label-{} not exists.".format(i))
            else:
                total_labels.remove(i)





def process():
    # data_dir = sys.argv[1]
    # source_data_dir = sys.argv[2]
    # dataset_name = sys.argv[3]
    # dataset_labels = sys.argv[4]

    for id, dataset_name in enumerate(dataset_name_list):
        one_data_dir = os.path.join(data_dir, dataset_name)
        one_source_data_dir = os.path.join(source_data_dir, dataset_name, "serialization", dataset_name)

        if os.path.exists(one_data_dir):
            print(dataset_name+" already exists, skiped!!!!!!!!!!!!!!!!!!!!!\n")
        else:
            print("copy {}".format(dataset_name))
            shutil.copytree(one_source_data_dir, one_data_dir)

    check_labels()


if __name__ == '__main__':
    process()




