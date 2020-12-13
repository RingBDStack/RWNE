"""
heyu
https://blog.csdn.net/pipisorry/article/details/37742423
http://python.jobbole.com/85106/
http://matplotlib.org/api/pyplot_api.html
http://www.jianshu.com/p/c495e663f0ed
http://blog.csdn.net/wizardforcel/article/details/54407212
http://www.cnblogs.com/wei-li/archive/2012/05/23/2506940.html
https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
http://www.cnblogs.com/darkknightzh/p/6117528.html
"""

import numpy as np
import os
import matplotlib.pyplot as plt, mpld3
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import gridspec



def get_classify_result(file):
    test_size_macro_micro = dict()
    test_size = '0'
    macro_f1 = 0
    micro_f1 = 0
    for line in open(file):
        line = line.strip()
        if line.startswith('test_size'):
            test_size_macro_micro[test_size] = [macro_f1, micro_f1]
            test_size = line.split('=')[-1].strip()
        elif line.startswith('Macro_F1'):
            macro_f1 = float(line.split('=')[-1].strip())
        elif line.startswith('Micro_F1'):
            micro_f1 = float(line.split('=')[-1].strip())
        else:
            continue
    test_size_macro_micro[test_size] = [macro_f1, micro_f1]

    macro_f1_list = []
    micro_f1_list = []
    for test_size in ['0.'+str(v) for v in range(9,0,-1)]:
        if test_size in test_size_macro_micro.keys():
            macro_f1_list.append(test_size_macro_micro[test_size][0])
            micro_f1_list.append(test_size_macro_micro[test_size][1])
        else:
            macro_f1_list.append(0)
            micro_f1_list.append(0)

    return macro_f1_list, micro_f1_list

def get_cluster_result(file):
    NMI = 0
    for line in open(file):
        line = line.strip()
        if line.startswith('NMI'):
            NMI = float(line.split('=')[-1].strip())
    return NMI


def get_best_micro(file):
    return float(open(file).readline().split(':')[1].strip())

def get_best_micros(exp):
    micros = []
    ckpt = open(os.path.join(exp, "best_ckpt.info")).readlines()[1].split(':')[1].strip()
    for line in open(os.path.join(exp, "classify.info")):
        if line.find(ckpt)>-1:
            for res in line.split(':')[-1].strip().split('    '):
                micros.append(float(res.split(',')[-1].strip()))
    return micros



def get_color_marker_linestyle(labels):
    # color_list = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
    #               '#fabebe', '#008080']
    color_list = ['m','r','b','g','#800000','c','#B8860B','#2F4F4F']
    linestyle_list = ['-', '--', '-.', ':']
    marker_list = ['+', 'o', 'D', '*']
    sub_exp_cms_list = []
    i = -1; j = -1
    for label in labels:
        if label.startswith('deepwalk'):
            sub_exp_cms_list.append(('k', '-', '^'))
        else:
            i += 1
            if i % len(color_list) == 0:
                j += 1
            sub_exp_cms_list.append((color_list[i % len(color_list)], linestyle_list[j], '^'))
    return sub_exp_cms_list


def subplot_accuracy_temporal_broken_line_graph(ax, Yarguments, labels):
    """
    绘制折线图
    :param Xargument_list: 横坐标、自变量
    :param Yargument_list: 纵坐标、因变量
    :return:
    """
    # 作出ggplot风格的图片
    # plt.style.use('ggplot')

    # 创建figure窗口，一般采用默认值
    # num:figure窗口编号
    # figsize:窗口大小,(width, height)
    # dpi: 分辨率，即清晰度
    # plt.figure(num=1, figsize=(10, 10))

    X = [v/10 for v in range(1,10)]
    styles = get_color_marker_linestyle(labels)
    lines = []

    # plot函数：
    # linestyle: 线条风格 	[‘solid’ | ‘dashed’, ‘dashdot’, ‘dotted’ | (offset, on-off-dash-seq)
    #                               | '-' | '--' | '-.' | ':' | 'None' | ' ' | '']
    # marker: 用什么来标记坐标点
    # color: 线条颜色
    # label: 为线条指定名称
    # linewidth:线条宽度
    # antialiased: 抗锯齿
    for exp in range(0, len(Yarguments)):
        lines += ax.plot(X, Yarguments[exp], color=styles[exp][0], linestyle=styles[exp][1], marker=styles[exp][2],
                         label=labels[exp], antialiased=True, linewidth=2.0, markersize=7)

    # 设置坐标范围
    plt.xlim((0, 1))

    # 显示次刻度标签的位置,没有标签文本
    # ax.yaxis.set_minor_locator(yminorLocator)

    my_x_ticks = np.arange(0, 1, 0.1)
    plt.xticks(my_x_ticks)

    plt.tick_params(axis='both', which='major', labelsize=18)

    # my_y_ticks = np.arange(0.9, 1, 0.005)
    # plt.yticks(my_y_ticks)

    ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5)  # y坐标轴的网格使用次刻度
    ax.yaxis.grid(True, which='minor', linestyle='--', linewidth=0.4)  # y坐标轴的网格使用次刻度

    return lines

def plot_classify_graph(Micro_arguments, labels, title, save_path=None):

    FONTSIZE = 18 #44
    fontdict = {'family': 'serif', 'size': FONTSIZE}

    fig = plt.figure()
    # specifies the location of the subplot in the figure
    gs = gridspec.GridSpec(1, 2, width_ratios=[4,1])
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])

    plt.axes(ax1)

    # 轴标签
    plt.xlabel('Train ratio', style='italic', fontdict=fontdict, fontsize = FONTSIZE)  # FONTSIZE - 4
    plt.ylabel('Micro_F1', style='italic', fontdict=fontdict, rotation=90, fontsize=FONTSIZE) # FONTSIZE - 4

    # plt.ylim((0.8, 1))

    # 设置坐标刻度，人为设置坐标轴的刻度显示的值，也可以显示文字作为坐标
    # ymajorLocator = MultipleLocator(0.01)  # 将y轴主刻度标签设置为0.01的倍数
    # ymajorFormatter = FormatStrFormatter('%1.2f')  # 设置y轴标签文本的格式
    # # 设置主刻度标签的位置,标签文本的格式
    # ax1.yaxis.set_major_locator(ymajorLocator)
    # ax1.yaxis.set_major_formatter(ymajorFormatter)
    lines = subplot_accuracy_temporal_broken_line_graph(ax1, Micro_arguments, labels)
    plt.setp(ax1.get_xticklabels(), fontsize=FONTSIZE) # FONTSIZE-10
    plt.setp(ax1.get_yticklabels(), fontsize=FONTSIZE) # FONTSIZE-10
    plt.title(title,fontsize=FONTSIZE+6)

    ax2.axis("off")
    leg = ax2.legend(tuple(lines), tuple(labels), loc='lower center', ncol=1)
    plt.setp(leg.texts, family='serif')
    # ax2.legend(tuple(lines), tuple(labels), loc='upper left', labelspacing=0.8, fontsize=16)

    # plt.subplots_adjust(hspace=0.1, wspace=0.18, bottom=0.25)  # wspace
    if save_path!= None:
        plt.savefig(os.path.join(save_path, 'plot_results.pdf'), format='pdf', bbox_inches='tight', dpi=200)
        plt.savefig(os.path.join(save_path, 'plot_results.png'), format='png', bbox_inches='tight', dpi=200)
    plt.show()
    # mpld3.show()



def generate_txt():
    data_path = "/home/heyu/work/result/NRL/"

    method =  "rwne_blogcatalog_P-v_embed-context"

    batch_size = (1,2,4,8,16,32,64,128)
    walk_length = (1,2,4,8,16,32,64,128)


    with open("result.txt",'w') as fr:
        fr.write("batch_size\\walk_length\n")
        for bs in batch_size:
            for wl in walk_length:
                micro = get_best_micro(os.path.join(data_path,method,"batch_size_%d_walk_length_%d"%(bs,wl), "best_ckpt.info"))
                fr.write("%.4f\t"%micro)
            fr.write("\n")


def plot_graph():
    data_path = "/home/heyu/work/result/NRL/"
    methods = (
        "rwne_blogcatalog_P-v_embed-context",
            )

    title = "mean micro for ratio 0.9"

    batch_size = (1,)
    walk_length = (1,2,4,8,16,32,64,128)

    # if len(methods) == 1:
    #     save_path = os.path.join(data_path, methods[0])
    # else:
    #     save_path = None

    exps = []
    labels = []
    for method in methods:
        for bs in batch_size:
            for wl in walk_length:
                labels.append(method[17:]+"/batch_size_%d_walk_length_%d"%(bs,wl))
                exps.append(os.path.join(data_path,method,"batch_size_%d_walk_length_%d"%(bs,wl)))

    Micro_arguments = []
    Micro_arguments.append(
        [0.3505, 0.3801, 0.3923, 0.4023, 0.4090, 0.4134, 0.4171, 0.4183, 0.4143]
    ) # insert result of deepwalk
    labels.insert(0,"deepwalk")
    for exp in exps:
        micro = get_best_micros(exp)
        Micro_arguments.append(micro)
        # nmi = get_cluster_result(os.path.join(data_path, exp, 'cluster', 'cluster.' + type))
        # micro.insert(0, nmi)

    plot_classify_graph(Micro_arguments, labels, title)


if __name__ == '__main__':
    generate_txt()
    # plot_graph()
