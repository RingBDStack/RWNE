#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.0.0 
 
"""


import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


# RWR:
# w(t)=(1+(L-t)(1-α))*α^t

# L_list = [2,5,10,20,50]
L_list = [10]
# a_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
a_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


fig = plt.figure()

gs = gridspec.GridSpec(1, 2)

i = 0
for i, l in enumerate(L_list):
    ax1 = plt.subplot(gs[i])
    i +=1
    plt.axes(ax1)

    # 轴标签
    plt.xlabel('x: t (order)')  # FONTSIZE - 4
    plt.ylabel('y: w (weight)') # FONTSIZE - 4


    lines = []
    labels = []
    for a in a_list:
        T = np.arange(1,l+1)
        Y = (1+(l-T)*a)*np.power(1-a,T)
        # Y = Y / np.sum(Y) #Y[0]
        Y = Y / Y[0]
        lines += plt.plot(T, Y)
        labels.append("restart prob={}".format(a))

    # plt.ylim((0.8, 1))

    # 设置坐标刻度，人为设置坐标轴的刻度显示的值，也可以显示文字作为坐标
    ymajorLocator = MultipleLocator(0.1)  # 将y轴主刻度标签设置为0.01的倍数
    ymajorFormatter = FormatStrFormatter('%1.1f')  # 设置y轴标签文本的格式
    # # 设置主刻度标签的位置,标签文本的格式
    ax1.yaxis.set_major_locator(ymajorLocator)
    ax1.yaxis.set_major_formatter(ymajorFormatter)
    plt.setp(ax1.get_xticklabels())  # FONTSIZE-10
    plt.setp(ax1.get_yticklabels())  # FONTSIZE-10
    yminorLocator = MultipleLocator(0.05)
    ax1.yaxis.set_minor_locator(yminorLocator)
    ax1.yaxis.grid(True, which='major', linestyle='--')  # y坐标轴的网格使用次刻度
    ax1.yaxis.grid(True, which='minor', linestyle='--')  # y坐标轴的网格使用次刻度
    plt.title("walk_length L = {}, restart prob.".format(l))



ax2 = plt.subplot(gs[i])
ax2.axis("off")
leg = ax2.legend(tuple(lines), tuple(labels), loc='lower center', ncol=2)
plt.setp(leg.texts, family='serif')

plt.subplots_adjust(hspace=0.5, wspace=0.2)
plt.show()
