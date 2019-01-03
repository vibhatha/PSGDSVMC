import numpy as np
import scipy as sp
import os
import sys

base_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/2018_12_31/parallel/ijcnn1_epochlog/"

list = os.listdir(base_path)
for file in list:
    parallelism = file.split('world_size_')[1].split('_')[0]
    comm_gap = file.split('comm_gap=')[1].split('_')[0]
    num_lines = sum(1 for line in open(base_path+file))
    #print(parallelism, comm_gap, num_lines)
    os.rename(base_path+file,base_path+"ijcnn_m="+str(parallelism)+"_c="+str(comm_gap)+"_i="+str(num_lines))