#! /usr/bin/env python  
# -*- coding:utf-8 -*-

"""
__author__ = "He Yu"
Version: 1.5.4
mainly reference:
    deepwalk:
        https://github.com/phanein/deepwalk
    python logging:
        http://www.jb51.net/article/42626.htm
        https://www.cnblogs.com/dkblog/archive/2011/08/26/2155018.html
        https://docs.python.org/3/library/logging.html#logging.Logger.setLevel
    argparse:
        http://blog.csdn.net/itlance_ouyang/article/details/52489674
"""

import sys
import logging
import traceback
import logging.handlers
import os
import time
import psutil
from multiprocessing import cpu_count
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import utils
import walker
import trainer
import eval_reconstruction
import eval_classify
import eval_cluster
import eval_visualization
import eval_link_prediction
import data_lp_split_network
import data_serialization


# global variables
logger = logging.getLogger('NRL')
# logger.setLevel(logging.NOTSET)
# Note: NOTSET level is not effective by default!!!
# If we set level by 'logger.setLevel(logging.NOTSET)' or 'logging.getLogger('')' or do not set level here,
# the logger will have a same level with its parent(i.e. root logger), for the term ‘delegation to the parent’.
# But, actually, the root logger is created with level WARNING, not the NOTSET, so we will create a logger with a
# level WARNING in fact. And for this, the handler's level-set of INFO in below is never going into effect.
logger.setLevel(logging.DEBUG) # Note: this step can't be omitted for the term ‘delegation to the parent’.

# DATA_PATH = '/home/heyu/work/data/HIN'



def set_cpu_affinity():
    """
    Set cpu affinity to run on multiple processes.
    https://www.cnblogs.com/liu-yao/p/5678157.html
    :return:
    """
    # Monitor current progress.
    p = psutil.Process(os.getpid())
    # Set cpu affinity to run on multiple processes.
    try:
        # p.set_cpu_affinity(list(range(cpu_count())))
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        # logger.exception("set_cpu_affinity failed.")
        logger.warning("set_cpu_affinity failed.")

def set_logging(options):
    """
    set logging
    :return:
    """

    console_numeric_level = getattr(logging, options.log.upper(), None)
    file_numeric_level = getattr(logging, options.log_for_file.upper(), None)
    if not isinstance(console_numeric_level, int):
        raise ValueError('Invalid log level: %s' % options.log)
    if not isinstance(file_numeric_level, int):
        raise ValueError('Invalid log level: %s' % options.log_for_file)

    log_format_file = '%(asctime)s : %(filename)s[line:%(lineno)d] : %(levelname)s : %(message)s'
    date_format = '%Y-%b-%d %H:%M:%S'
    log_format_console = '[%(levelname)-8s] : %(message)s'

    if not os.path.exists(os.path.split(options.log_name)[0]):
        os.makedirs(os.path.split(options.log_name)[0])
    # fh = logging.FileHandler(options.log_name, mode='a')
    fh = logging.handlers.RotatingFileHandler(options.log_name,mode='a',
                                              maxBytes=50 * 1024 * 1024, # 100M
                                              backupCount = 5)
    # if console_numeric_level < logging.INFO:
    #     fh.setLevel(console_numeric_level)
    # else:
    #     fh.setLevel(logging.INFO)
    fh.setLevel(file_numeric_level)

    ch = logging.StreamHandler()
    ch.setLevel(console_numeric_level)

    formatter_file = logging.Formatter(fmt=log_format_file,datefmt=date_format)
    fh.setFormatter(formatter_file)
    if options.log_full:
        ch.setFormatter(formatter_file)
    else:
        formatter_cons = logging.Formatter(fmt=log_format_console)
        ch.setFormatter(formatter_cons)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # this is an imposing separation line.
    separa_line = ('\n' + '-' * 100)
    logger.info(separa_line*4)
    logger.info(separa_line * 2 + '\nlog configured, new running starts...' + separa_line * 2)

def parse_args():
    """ parse arguments."""
    #################################################################
    parser = ArgumentParser("NRL, Networok Representation Learning",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    ################################################################

    # args to build the whole model
    parser.add_argument("-d","--description", dest="description",
                        help="a special descriptive sentence to mark this running.")
    parser.add_argument("--task", dest="task", nargs="*", default=["train"],
                        choices=["walk", "train", "classify", "cluster", "reconstruction", "link_prediction", "visualization",
                                 "split", "serialization"],
                        help="choosing runing task(s):\n\t\t"
                             "walk: only walk a training corpus on a netwrok (for walk based models);\n\t\t"
                             "train: train vectors from walks files (without walk), if files not exist, walk task will auto start;\n\t\t"
                             "classify: perform classification application task using trained vectors, if vectors not exist, train task will auto start.\n\t\t"
                             "cluster: perform cluster application task using trained vectors.\n\t\t"
                             "reconstruction: perform graph reconstruction task.\n\t\t"
                             "visualization: perform visualization task using matplotlib.\n\t\t"
                             "link_prediction: perform link-prediction task by division.\n\t\t"
                             "split: data pre-process. split the network by edges.\n\t\t"
                             "serialization: sort nodesID, make them consistent.")
    parser.add_argument("--model", dest="model",default="DeepWalk",
                        choices=["RWNE","DeepWalk","Node2Vec","LINE","SDNE","DNGR","GraRep","GCN"],
                        help="the optional NRL models.")
    parser.add_argument("-tf", "--using_tensorflow", dest="using_tensorflow", default=False, action='store_true',
                        help="training in tensorflow.")
    parser.add_argument("--log", dest="log", default="INFO",
                        help="log verbosity level for console:CRITICAL,ERROR,WARNING,INFO,DEBU,GNOTSET")
    parser.add_argument("--log_for_file", dest="log_for_file", default="INFO",
                        help="log verbosity level for file:CRITICAL,ERROR,WARNING,INFO,DEBU,GNOTSET")
    parser.add_argument("--log_name", dest="log_name", default="./log/run.log",
                        help="log file name.")
    parser.add_argument("-lf", "--log_full", dest="log_full", default=False, action='store_true',
                        help="print a full-format log info to console.")
    parser.add_argument("-ar","--always_rebuild", dest="always_rebuild", default=False, action='store_true',
                        help="always rebuild when files exits (like corpus, vectors, ...).")
    parser.add_argument("--re_direction_path", default="./log/std.log",
                        help="redirect the standard output and error info.")

    # parameters for walk
    parser.add_argument("--data_path", dest="data_path", default="./data/karate.adjlist",
                        help="the input network data path.")
    parser.add_argument("--data_format", dest="data_format", default="adjlist", choices=["adjlist", "edgelist"],
                        help="data format of input network.")
    parser.add_argument("-dir","--isdirected", dest="isdirected", default=False, action='store_true',
                        help="the edge is directed or undirected.")
    parser.add_argument("--net_info_path", dest="net_info_path", default="./tmp/net/net.info",
                        help="filepath to record network info.")
    parser.add_argument("--walk_times", dest="walk_times", default=10, type=int,   # 100
                        help="number of random walks to start at each node.")
    parser.add_argument("--walk_length", dest="walk_length", default=80, type=int,   # 100 # 25
                        help="the length of each random walk started at each node.")
    parser.add_argument("--walk_restart", dest="walk_restart", default=0.5, type=float,  # 100
                        help="random walk with restart.")
    # parser.add_argument("-using_restart_walk", dest="using_restart_walk", default=False, action='store_true',
    #                     help="whether using the restart walk by walk_restart control.")
    # parser.add_argument("--walk_strategy", dest="walk_strategy", default="shuffle", choices=["uniform", "unigram", "shuffle"],
    #                     help="the strategy to select 'batch_size' restart_random_walk nodes to generate training batch.")
    parser.add_argument('--p', dest="p", type=float, default=1,
                        help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', dest="q", type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')
    parser.add_argument("--walk_workers", dest="walk_workers", default=cpu_count(), type=int,
                        help="speed up walking by using multi-process (by set as 0 or 1 to close multi-process optimizer).")
    parser.add_argument("-wtm", "--walk_to_memory", dest="walk_to_memory", default=False, action='store_true',
                        help="if true, walk all sectences in memory;\n"
                             "if false, walk streaming sentences and write them to files online.")
    parser.add_argument("-nsc", "--not_store_corpus", dest="not_store_corpus", default=False, action='store_true',
                        help="decide to whether store walks corpus, only valid when walk_to_memory is true.")
    parser.add_argument("-csp", "--corpus_store_path", dest="corpus_store_path", default="./tmp/walks/corpus.walks",
                        help="the path to store walks corpus.")


    # parameters for train
    parser.add_argument('--iter_epoches', dest="iter_epoches", default=1000, type=float,
                        help='iter_epoches for train')
    parser.add_argument('--embedding_size', dest="embedding_size", default=128, type=int,
                        help='the dimension of embedding vector to learn for each node.')
    parser.add_argument('--window_size', dest="window_size", default=10, type=int,
                        help='the window size of skip-gram model.')
    parser.add_argument('--negative', dest="negative", default=5, type=int,
                        help='negative sampling per node of skip-gram model.')
    parser.add_argument('--downsample', dest="downsample", default=1e-3, type=float,
                        help='The threshold for configuring which higher-frequency words are randomly downsampled, '
                             'no words downsampled if setting the value as 0.')
    parser.add_argument('--distortion_power', dest="distortion_power", default=0.75, type=float,
                        help='distort the neg-sampling unigrams.')
    parser.add_argument('-lr','--learning_rate', dest="learning_rate", default=0.01, type=float,
                        help='initial learn rate.')
    parser.add_argument('--order', dest="order", default = "3", choices = ["1", "2", "3"],
                        help='order for LINE mode.')
    parser.add_argument('--train_workers', dest="train_workers", default=cpu_count(), type=int,
                        help='speed up training by using multi-thread.')
    parser.add_argument("-ctl", "--close_train_log", dest="close_train_log", default=False, action='store_true',
                        help="close the log info(train details) of called word2vec module.")
    parser.add_argument("-lfm", "--load_from_memory", dest="load_from_memory", default=False, action='store_true',
                        help="train skip-gram model from memory(load all corpus to memory)")
    parser.add_argument("--model_path", dest="model_path", default="./tmp/vec/embedding.model",
                        help="the path to store trained model.")
    parser.add_argument("--vectors_path", dest="vectors_path", default="./tmp/vec/embedding.vectors",
                        help="the path to store trained vectors.")
    parser.add_argument("--unshuffled", dest="unshuffled", default=False, action='store_true',
                        help="indicater to shuffle examples during training process.")
    parser.add_argument("--decay_epochs", dest="decay_epochs", default=0, type=float,  # 100000
                        help="decay the learning rate every decay_epochs;\n"
                             "note that decay_epochs must be a integer > 0 (0 means learning without decay, "
                             "which is also a default chioce).")
    parser.add_argument("--decay_interval", dest="decay_interval", default=600, type=int)
    parser.add_argument("--decay_rate", dest="decay_rate", default=0.1, type=float,
                        help="decay_rate for exponentially decay.")
    parser.add_argument("--loss_interval", dest="loss_interval", default=10, type=int)
    parser.add_argument("--summary_steps", dest="summary_steps", default=1000, type=int)
    parser.add_argument("--summary_interval", dest="summary_interval", default=60, type=int)
    parser.add_argument("--ckpt_interval", dest="ckpt_interval", default=3600, type=int,
                        help="interval(seconds) to record checkpoint files.")
    parser.add_argument("--ckpt_epochs", dest="ckpt_epochs", default=1, type=float,
                        help="epochs to record checkpoint files.")
    parser.add_argument("--batch_size", dest="batch_size", default=8, type=int,  # 1000000
                        help="batch size for SGD.")
    parser.add_argument("--local_weight", dest="local_weight", default=0.5, type=float,  # 1000000
                        help="the weight for MSE loss.")
    parser.add_argument("--global_weight", dest="global_weight", default=0.5, type=float,  # 1000000
                        help="the weight for MSE loss.")

    parser.add_argument("--dbn_initial", dest="dbn_initial", default=False, action='store_true',
                        help="dbn run or not")
    parser.add_argument("--dbn_epochs", dest="dbn_epochs", default=100, type=int,  # 1000000
                        help="dbn_epoch")
    parser.add_argument("--dbn_batchsize", dest="dbn_batchsize", default=64, type=int,  # 1000000
                        help="dbn_batchsize.")
    parser.add_argument("--dbn_learning_rate", dest="dbn_learning_rate", default=0.01, type=float,  # 1000000
                        help="dbn_learning_rate")
    parser.add_argument("--active_function", dest="active_function", default="sigmoid",
                        choices=["sigmoid","tanh","relu","leaky_relu"],
                        help="the active_function of neural units, choices:sigmoid, tanh, relu, leaky_relu.")
                        

    parser.add_argument("--struct", dest="struct", type=int, nargs="*", default=[],
                        help="struct for SDNE.")
    parser.add_argument("--alpha", dest="alpha", default=100, type=float,
                        help="weighing hyperparameter for 1st order objective in SDNE model.")
    parser.add_argument("--beta", dest="beta", default=10, type=float,
                        help="penalty parameter in matrix B of 2nd order objective in SDNE model.")
    parser.add_argument("--gamma", dest="gamma", default=1, type=float,
                        help="weighing hyperparameter for 2nd order objective in SDNE model.")
    parser.add_argument("--reg", dest="reg", default=1, type=float,
                        help="L2-reg hyperparameter in SDNE model.")
    parser.add_argument("--sparse_dot", dest="sparse_dot", default=False, action='store_true')


    parser.add_argument("-ldp","--log_device_placement", dest="log_device_placement", default=False, action='store_true',
                        help="whether device placements should be logged.")
    parser.add_argument("-asp", "--allow_soft_placement", dest="allow_soft_placement", default=False,
                        action='store_true',
                        help="whether soft placement is allowed.")
    parser.add_argument("-gmf", "--gpu_memory_fraction", dest="gpu_memory_fraction",
                        default=1.0, type=float,
                        help="a value between 0 and 1 that indicates what fraction of the available GPU memory "
                             "to pre-allocate for each process.")
    parser.add_argument("-ag", "--allow_growth", dest="allow_growth", default=False, action='store_true',
                        help="if true, the allocator does not pre-allocate the entire specified GPU memory region,"
                             " instead starting small and growing as needed.")
    parser.add_argument("-gpu", "--using_gpu", dest="using_gpu", default=False, action='store_true',
                        help="whether using gpu.")
    parser.add_argument("--gpu_devices",dest="visible_device_list", type=int, nargs="*", default=[0],
                        help="a list of GPU ids that determines the 'visible' to 'virtual' mapping of GPU devices.")


    ### gcn
    parser.add_argument("-using_label", "--using_label", dest="using_label", default=False, action='store_true',
                        help="whether using_label when training, i.e., supervised?")
    parser.add_argument("--feature_type", dest="feature_type", default="random",
                        choices=["attribute", "random", "degree", "adjacency"])
    parser.add_argument("--feature_size", dest="feature_size", default=128, type=int,
                        help="the feature_size.")
    parser.add_argument("--feature_path", dest="feature_path", default="./data/karate.features",
                        help="the data feature path.")
    parser.add_argument("--dropout", dest="dropout", default=0., type=float,
                        help="Dropout rate (1 - keep probability)")
    parser.add_argument("--weight_decay", dest="weight_decay", default=5e-4, type=float,
                        help="weight_decay for l2_norm loss.")
    parser.add_argument("--hiddens_size", dest="hidden_size_list", nargs="*", default=[128], type=int,
                        help="hidden_size in Graph Convolution Layer")

    # parameters for evaluate
    parser.add_argument("--label_path", dest="label_path", default="./data/karate.labels",
                        help="the data label path.")
    parser.add_argument("--label_size", dest="label_size", default=0, type=int,
                        help="the total labels size.")
    parser.add_argument("-eval_online", dest="eval_online", default=False, action='store_true',
                        help="Whether to run eval on-line, if False, run the eval only once.")
    parser.add_argument("--eval_interval", dest="eval_interval", default=300, type=int,
                       help="eval_interval (secs): How often to run the eval.")
    parser.add_argument("--repeated_times", dest="repeated_times", default=10, type=int,
                        help="eval repeated_times")
    parser.add_argument("--train_ratio", dest="train_ratio", default=0, type=float,
                        help="train_ratio to split the original data for classify task and link_prediction task: if > 0 means a specified ratio; else means a default ratio list from 0.1 to 0.9.")
    parser.add_argument('--eval_workers', dest="eval_workers", default=cpu_count(), type=int,
                        help='speed up evaling by using multi-thread.')
#    ## LogisticRegression(solver=options.solver, n_jobs=options.n_jobs)
#    parser.add_argument("--solver", dest="solver", default="liblinear",
#                        help="the solver for LogisticRegression: 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'")
#    parser.add_argument("--n_jobs", dest="n_jobs", default=1, type=int,
#                        help="Number of CPU cores used when parallelizing over classes if multi_class='ovr'."
#                             "If given a value of -1, all cores are used.")
    parser.add_argument("--precK_max_index",dest="precK_max_index",default=10000,type=int,
                        help="the max_index value in metric precise@K for reconstruction and link-prediction.")
    parser.add_argument("--sample_nodes", dest="sample_nodes", default=0, type=int,
                        help="sample some nodes to reduce computation for reconstruction and link-prediction.\n"
                             "if > 0 means a specified sample number; else means using the whole nodes set.")
    parser.add_argument("--sample_nodes_rule", dest="sample_nodes_rule", default="extend", choices=["random", "extend"],
                        help="the rule to sample nodes for reconstruction and link-prediction.\n\t\t"
                             "random: randomly sample all sampled_nodes;\n\t\t"
                             "extend: randomly sample one root-node and then extend to sampled_nodes.")
    parser.add_argument("--similarity_metric", dest="similarity_metric", default="cosine", choices=["cosine", "euclidean"],
                        help="similarity_metric: cosine, euclidean.")
    parser.add_argument("--marker_size",dest="marker_size",default=10, type=int,
                        help="the plot marker size for visualization.")
    parser.add_argument("--multilabel_rule", dest="multilabel_rule", default="ignore", choices = ["first", "random", "ignore", "all"],
                        help="the rule to handler multi-label nodes for visualization task and cluster task.\n\t\t"
                             "first: always choose the first label as the normal label for multi-label nodes.\n\t\t"
                             "random: randomly choose a label as the normal label for multi-label nodes.\n\t\t"
                             "ignore: ignore the multi-label nodes.\n\t\t"
                             "all: choose all labels.")
    parser.add_argument("--classify_path", dest="classify_path", default="./tmp/classify/classify.info",
                        help="the eval results for classify.")
    parser.add_argument("--cluster_path", dest="cluster_path", default="./tmp/cluster/cluster.info",
                        help="the eval results for cluster.")
    parser.add_argument("--reconstruction_path",dest="reconstruction_path",default="./tmp/reconstruction/reconstruction.info",
                        help="the reconstruction info and results.")
    parser.add_argument("--visualization_path",dest="visualization_path",default="./tmp/visualiazation/visualiazation.info",
                        help="the eval results for visualiazation.")
    parser.add_argument("--link_prediction_path",dest="link_prediction_path",default="./tmp/link_prediction/link_prediction.info",
                        help="eval results for link_prediction")
    parser.add_argument("--eval_data_path", dest="eval_data_path", default="./splited_eval_0.2_repeat_1/blogcatlog.adjlist",
                        help="the eval data_path for link_prediction task.")
    parser.add_argument("--except_data_path", dest="except_data_path", default="./splited_train_0.8_repeat_1/blogcatlog.adjlist",
                        help="the except data_path (i.e., the actual train data_path) for link_prediction task.")


    # for data preprocess:
    parser.add_argument("--repeat_from", dest="repeat_from", default=1, type=int)
    parser.add_argument("--repeat_to", dest="repeat_to", default=10, type=int)
    parser.add_argument("--target_data_dir", dest="target_data_dir", default="./sorted")
    parser.add_argument("--keep_isolated", dest="keep_isolated", default=False, action='store_true')


    args = parser.parse_args()

    # dependencies
    # abs path
    args.log_name = os.path.abspath(args.log_name)
    args.re_direction_path = os.path.abspath(args.re_direction_path)
    args.data_path = os.path.abspath(args.data_path)
    args.net_info_path = os.path.abspath(args.net_info_path)
    args.corpus_store_path = os.path.abspath(args.corpus_store_path)
    args.model_path = os.path.abspath(args.model_path)
    args.vectors_path = os.path.abspath(args.vectors_path)
    args.label_path = os.path.abspath(args.label_path)
    args.reconstruction_path = os.path.abspath(args.reconstruction_path)
    args.classify_path = os.path.abspath(args.classify_path)
    args.cluster_path = os.path.abspath(args.cluster_path)
    args.visualization_path = os.path.abspath(args.visualization_path)
    args.link_prediction_path = os.path.abspath(args.link_prediction_path)
    # to identify an index file when many files are generated
    args.headflag_of_index_file = '[!!!this is an index file[FILE: xxx]!!!]'  # FILE:

    return args

def process(options):
    logger.info('RUN description: {}'.format(options.description))
    logger.info('task: {}'.format(options.task))
    logger.info('model: {}'.format(options.model))

    with open(os.path.join(os.path.split(options.log_name)[0], "run.describ"),'w') as fr:
        fr.write('RUN description: {}\n'.format(options.description))
        fr.write('task: {}\nmodel: {}'.format(options.task,options.model))

    # preprocess:
    if "split" in options.task:
        data_lp_split_network.process(options)
    if "serialization" in options.task:
        data_serialization.process(options)

    # walk:
    walk_corpus = None
    
    if 'walk' in options.task:
        walk_corpus = walker.build_walk_corpus(options)
    # train
    if 'train' in options.task:
        trainer.train_vectors(options, sens=walk_corpus)
        open(os.path.join(os.path.split(options.vectors_path)[0], "RUN_SUCCESS"), 'w')

    # eval case:
    if 'classify' in options.task:
        eval_classify.eval(options)
        open(os.path.join(os.path.split(options.classify_path)[0], "RUN_SUCCESS"), 'w')

    if 'cluster' in options.task:
        eval_cluster.eval(options)
        open(os.path.join(os.path.split(options.cluster_path)[0], "RUN_SUCCESS"), 'w')

    if 'reconstruction' in options.task:
        eval_reconstruction.eval(options)
        open(os.path.join(os.path.split(options.reconstruction_path)[0], "RUN_SUCCESS"),'w')

    if 'visualization' in options.task:
        eval_visualization.eval(options)
        open(os.path.join(os.path.split(options.visualization_path)[0], "RUN_SUCCESS"), 'w')

    if "link_prediction" in options.task:
        eval_link_prediction.eval(options)
        open(os.path.join(os.path.split(options.link_prediction_path)[0],"RUN_SUCCESS"),'w')

def main():

    # parse argus
    args = parse_args()
    # redirect
    if len(os.path.split(args.re_direction_path)[1]) > 0:
        utils.redirect_stdinfo(args.re_direction_path)
    # set logging
    set_logging(args)
    # set cpu affinity
    set_cpu_affinity()

    time_start = time.time()
    
    process(args)

    logger.info('Program finished in {}s.'.format(time.time() - time_start))
    # this is an imposing separation line.
    separa_line = ('\n' + '-' * 100)
    logger.info(separa_line * 2 + '\nthis is the end of this running' + separa_line * 2)







if __name__ == "__main__":
    try:
        # sys.exit(main())
        main()
    except:
        # record error info into log:
        # traceback.print_exc()  # note it is superfluous if we decide log 'traceback.format_exc()'.
        logger.exception('Uncaught exception:', exc_info=traceback.format_exc())
        # traceback.print_exc(file=fr) # redirect error info.