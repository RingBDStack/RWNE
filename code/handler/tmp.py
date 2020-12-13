#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.0.0 
 
"""

import numpy as np
import os
import matplotlib.pyplot as plt







#
# # 数据个数
# n = 100
# # 均值为0, 方差为1的随机数
# x = np.random.normal(0, 1, n)
# y = np.random.normal(0, 1, n)
# c = ["yellowgreen" for _ in range(n)]
#
#
#
#
#
# # 计算颜色值
# color = np.arctan2(y, x)
#
#
# # 绘制散点图
# plt.scatter(x, y, s = 75, c = c, alpha = 0.5)
# # # 设置坐标轴范围
# # plt.xlim((-1.5, 1.5))
# # plt.ylim((-1.5, 1.5))
# # 不显示坐标轴的值
# plt.xticks(())
# plt.yticks(())
# # # 关闭坐标轴
# # plt.axis('off')
# plt.show()




embed = np.array([[v,v,v,v,v] for v in range(4)], dtype=np.float32)
nce_w_true = np.array([[v,v,v,v,v] for v in range(100,301,100)], dtype=np.float32)
nce_b_true = np.array([1,2,3],dtype=np.float32)
nce_w_true = np.transpose(nce_w_true)
true_logits = np.matmul(embed, nce_w_true)
print("true_logits=\n{}\nwith shape={}".format(true_logits, np.shape(true_logits)))
true_logits = true_logits + nce_b_true # 2*4
print("true_logits_add_b=\n{}\nwith shape={}".format(true_logits, np.shape(true_logits)))
true_xent = -np.log(1-1/(1+(np.exp(-true_logits))))
print("true_xent=\n{}\nwith shape={}".format(true_xent, np.shape(true_xent)))

print("\n====================\n")

embed = np.array([[v,v,v,v,v] for v in range(4,8)], dtype=np.float32)
nce_w_true = np.array([[v,v,v,v,v] for v in range(400,601,100)], dtype=np.float32)
nce_b_true = np.array([11,22,33],dtype=np.float32)
nce_w_true = np.transpose(nce_w_true)
true_logits = np.matmul(embed, nce_w_true)
print("true_logits=\n{}\nwith shape={}".format(true_logits, np.shape(true_logits)))
true_logits = true_logits + nce_b_true # 2*4
print("true_logits_add_b=\n{}\nwith shape={}".format(true_logits, np.shape(true_logits)))
true_xent = -np.log(1-1/(1+(np.exp(-true_logits))))
print("true_xent=\n{}\nwith shape={}".format(true_xent, np.shape(true_xent)))





