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
import network


logger = logging.getLogger("NRL")



def process(options):
    logger.info("Data preprocessing: network serialization ...")
    time_start = time.time()
    source_data_dir, data_filename = os.path.split(options.data_path)
    data_prefixname = data_filename.split(".")[0]
    data_format = options.data_format
    isdirected = options.isdirected
    target_data_dir = options.target_data_dir

    logger.info("\t source_data_dir = {}".format(source_data_dir))
    logger.info("\t data_filename = {}".format(data_filename))
    logger.info("\t data_format = {}".format(data_format))
    logger.info("\t isdirected = {}".format(isdirected))
    logger.info("\n\t target_data_dir = {}".format(target_data_dir))

    net = network.construct_network(data_path = options.data_path,
                              data_format = data_format,
                              net_info_path = os.path.join(source_data_dir, "net.info"),
                              isdirected = isdirected,
                              print_net_info = True)
    net.make_consistent(remove_isolated = not options.keep_isolated)
    target_data_format = ["adjlist", "edgelist"]
    for save_format in target_data_format:
        net.save_network(target_data_dir, data_prefixname, save_format)

    # for label
    source_data_path = os.path.join(source_data_dir, data_prefixname + ".labels")
    target_data_path = os.path.join(target_data_dir, data_prefixname + ".labels")
    if os.path.exists(source_data_path):
        with open(target_data_path, "w") as fr:
            for line in open(source_data_path):
                line = line.strip()
                if line:
                    linelist = line.split('\t')
                    source_id = int(linelist[0].strip())
                    if source_id in net._nodes_id:
                        fr.write("{}".format(net._nodes_id[source_id]))
                        for label in linelist[1:]:
                            fr.write("\t{}".format(label.strip()))
                        fr.write("\n")
    logger.info('Data preprocessing: network serialization completed in {}s.'.format(time.time() - time_start))







