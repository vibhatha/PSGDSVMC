import numpy as np
import os
import sys
from numpy import genfromtxt


def load_file_np(filename):
    print(filename)

def comms_comp_average(parallelism=2):
    base_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/2018_12_31/parallel/pegasos/ijcnn1/commscomp-2019-01-01/ijcnn1_comms_comp_2019_01_01/refactored/"#"/home/vibhatha/Documents/Research/logs/psgsvmc/2018_12_31/parallel/pegasos/ijcnn1/commscomp-2018-12-31/summary/"
    result_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/2018_12_31/parallel/pegasos/ijcnn1/commscomp-2019-01-01/ijcnn1_comms_comp_2019_01_01/results/"#"/home/vibhatha/Documents/Research/logs/psgsvmc/2018_12_31/parallel/pegasos/ijcnn1/commscomp-2018-12-31/result/"
    all_files = os.listdir(base_path)
    #print(all_files)
    list = []
    for file in all_files:
        par_prefix = "m="
        param = par_prefix + str(parallelism)
        if (param in file):
            list.append(file)

    #list.sort()
    print(list)
    comm_gaps = ['1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024', '2048', '4096']
    sorted_list = []
    for comm_gap in comm_gaps:
        comm_gap = 'c=' + comm_gap + '_'
        for file in list:
            if(comm_gap in file):
                sorted_list.append(file)

    print(sorted_list)

    # para
    pars = np.array([1,2,4,8,16,32,64,128,256,512,1024,2048,4096])
    # loading files and calculating the mean
    comms_comp_avg_times = []
    comms_comp_all_times = []
    for file in sorted_list:
        my_data = genfromtxt(base_path+file, delimiter=',')
        average = np.average(my_data,axis=0)
        sum = np.sum(my_data, axis=0)
        comms_comp_avg_time = average[1:]
        comms_comp_avg_times.append(comms_comp_avg_time)
        comms_comp_all_time = sum[1:]
        comms_comp_all_times.append(comms_comp_all_time)
    result_file = result_path + "comm_comp_average_time_m=" + str(parallelism) + "_all.csv"
    np.savetxt(result_file, np.array(comms_comp_avg_times), delimiter=',')
    result_file_cm = result_path + "comm_comp_totaltime_m=" + str(parallelism) + "_all.csv"
    np.savetxt(result_file_cm, np.array(comms_comp_all_times), delimiter=',')

pars = [4,8,16,32]
for p in pars:
    comms_comp_average(parallelism=p)