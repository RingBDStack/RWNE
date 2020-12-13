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
import eval_utils
import utils


logger = logging.getLogger("NRL")



def process(options):
    # parser.add_argument("--train_data_dir", dest="train_data_dir", default="./splited_train_0.8_repeat_1",
    #                     help="the train network data path. like: origin_data_dir/splited_train_${train_ratio}_repeat_${repeat_th}")
    # parser.add_argument("--eval_data_dir", dest="eval_data_dir", default="./splited_eval_0.2_repeat_1",
    #                     help="the eval network data path. like: origin_data_dir/splited_eval_${eval_ratio}_repeat_${repeat_th}")
    logger.info("Data preprocessing: network split ...")
    time_start = time.time()
    origin_data_dir, data_filename = os.path.split(options.data_path)
    data_prefixname = data_filename.split(".")[0]
    data_format = options.data_format
    isdirected = options.isdirected
    if options.train_ratio > 0:
        train_ratio_list = [options.train_ratio]
    else:
        train_ratio_list = [v / 10.0 for v in range(9, 0, -1)]


    logger.info("\t origin_data_dir = {}".format(origin_data_dir))
    logger.info("\t data_filename = {}".format(data_filename))
    logger.info("\t data_format = {}".format(data_format))
    logger.info("\t isdirected = {}".format(isdirected))
    logger.info("\t train_ratio = {}".format(train_ratio_list))
    logger.info("\t repeat_from = {}".format(options.repeat_from))
    logger.info("\t repeat_to = {}".format(options.repeat_to))
    logger.info("\t log_name = {}".format(options.log_name))
    logger.info("\t re_direction_path = {}".format(options.re_direction_path))


    net_origin = network.construct_network(data_path=options.data_path,
                                           data_format=data_format,
                                           print_net_info=False,
                                           isdirected=isdirected)


    for train_ratio in train_ratio_list:
        for repeat_th in range(options.repeat_from, options.repeat_to+1):
            logger.info("\ntrain_ratio = {}, repeat_th = {}, spliting ...".format(train_ratio, repeat_th))
            # train_data_dir = os.path.join(origin_data_dir, "splited_train_{}_repeat_{}".format(train_ratio, repeat_th))
            # eval_data_dir = os.path.join(origin_data_dir, "splited_eval_{}_repeat_{}".format(round(1-train_ratio, 1), repeat_th))
            train_data_dir = os.path.join(origin_data_dir, "splited_train_{}_repeat_{}_train".format(train_ratio, repeat_th))
            eval_data_dir = os.path.join(origin_data_dir, "splited_train_{}_repeat_{}_eval".format(train_ratio, repeat_th))
            logger.info("\t train_data_dir = {}".format(train_data_dir))
            logger.info("\t eval_data_dir = {}".format(eval_data_dir))
            if os.path.exists(os.path.join(train_data_dir, data_prefixname + "." + data_format)):
                logger.info("train_ratio = {}, repeat_th = {}, already splited, skiped!!!\n".format(train_ratio, repeat_th))
                continue

            if not os.path.exists(train_data_dir):
                os.mkdir(train_data_dir)
            if not os.path.exists(eval_data_dir):
                os.mkdir(eval_data_dir)

            net_train, net_eval = net_origin.split_by_edges(train_ratio=train_ratio)
            net_train.save_network(train_data_dir, data_prefixname, data_format)
            # net_train.print_net_info(edges_file=os.path.join(train_net_dir, datafilename), file_path=os.path.join(train_net_dir, "net.info"))
            net_eval.save_network(eval_data_dir, data_prefixname, data_format)
            # net_eval.print_net_info(edges_file=os.path.join(eval_net_dir, datafilename), file_path=os.path.join(eval_net_dir, "net.info"))

            # eval_utils.split_network(origin_net_dir = origin_data_dir,
            #                       train_net_dir = train_data_dir,
            #                       eval_net_dir = eval_data_dir,
            #                       data_prefixname = data_prefixname,
            #                       data_format = data_format,
            #                       isdirected = isdirected,
            #                       train_ratio = train_ratio)
            logger.info("train_ratio = {}, repeat_th = {}, split succssed!!!\n".format(train_ratio, repeat_th))

    logger.info('Data preprocessing: network split completed in {}s.'.format(time.time() - time_start))


