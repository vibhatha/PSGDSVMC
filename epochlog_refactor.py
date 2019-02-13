import numpy as np
import scipy as sp
import os
import sys

# This script must be used to convert the cross-validation accuracy based epoch logs to a readable log format
# This script is used to convert the logs in  "PSGDSVMC/logs/epochlogs/parallel/pegasos/batch" to a readable format.

def epoch_cost():
    base_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/2018_12_31/parallel/pegasos/ijcnn1/commscomp-2018-01-03/ijcnn1/"#"/home/vibhatha/Documents/Research/logs/psgsvmc/2018_12_31/parallel/ijcnn1_epochlog/"
    list = os.listdir(base_path)
    for file in list:
        parallelism = file.split('world_size=')[1].split('_')[0]
        comm_gap = file.split('comm_gap=')[1].split('_')[0]
        num_lines = sum(1 for line in open(base_path+file))
        #print(parallelism, comm_gap, num_lines)
        os.rename(base_path+file,base_path+"ijcnn_m="+str(parallelism)+"_c="+str(comm_gap)+"_i="+str(num_lines))

def epoch_cost_2():
    base_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/2018_12_31/parallel/pegasos/ijcnn1/epochlogs-2018-01-03/temp_2/ijcnn_epochlog_parallel_m=2/ijcnn1/"
    base_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/2018_12_31/parallel/pegasos/ijcnn1/commscomp-2018-01-03/ijcnn1/"
    base_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/2019_01_09/ijcnn1_comms_m=16_2019_01_09/ijcnn1/"
    base_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/2019_01_18/2019_01_22_epsilon_epochlog/epsilon/" # working link
    base_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/2019_01_25/parallel/epsilon/epsilon/"
    base_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/2019_02_12/epsilon/5001/"
    list = os.listdir(base_path)
    for file in list:
        parallelism = file.split('world_size_')[1].split('_')[0]
        comm_gap = file.split('comm_gap=')[1].split('_')[0]
        num_lines = sum(1 for line in open(base_path+file))
        #print(parallelism, comm_gap, num_lines)
        if(num_lines==5001):
            os.rename(base_path+file,base_path+"ijcnn_m="+str(parallelism)+"_c="+str(comm_gap)+"_i="+str(num_lines))

epoch_cost_2()