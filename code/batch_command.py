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
import sys

def nohup_bash_repeat_from_to(bash_name = "lp_node2vec_repeat_from_to_flickr.sh",
                              train_ratio=0,
                              repeat_from = 1,
                              repeat_to = 10,
                              ratio_repeat_from = 1,
                              ratio_repeat_to = 10,
                              ):
    name = bash_name.split(".")[0].split("_")[-1]
    command = "nohup bash {0} {1} {2} {3} {4} {5} > mylog.{6}{1}_{4}_{5}_{2}_{3} 2>&1 &".format(bash_name, train_ratio, ratio_repeat_from, ratio_repeat_to, repeat_from, repeat_to, name)
    print(command)
    os.system(command)
    time.sleep(2)


def main():
    bash_name = sys.argv[1]
    train_ratio = sys.argv[2]
    ratio_repeat_from = sys.argv[3]
    ratio_repeat_to = sys.argv[4]
    for ratio_repeat in range(int(ratio_repeat_from.strip()), int(ratio_repeat_to.strip())+1):
        nohup_bash_repeat_from_to(bash_name=bash_name,
                                  train_ratio = train_ratio,
                                  repeat_from=1, repeat_to=1,
                                  ratio_repeat_from=ratio_repeat, ratio_repeat_to=ratio_repeat)


if __name__ == '__main__':
    main()