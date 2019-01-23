# this script allows you to sort out the the process 0 or any process out of the comm_comp logs

import os
import sys
import numpy as np
from numpy import genfromtxt


base_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/2019_01_12/webspam/all/group_comms_comp/all/"
result_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/2019_01_12/webspam/all/group_comms_comp/results/"

def sort_proc_comm_comp_logs(dataset="ijcnn1",proc_id=0, world_size=2, comm_gap=1, epochs=5001):
    list = os.listdir(base_path)
    proc_sorted = []
    str_world_size = "_world_size=" + str(world_size)+"__"
    str_proc_id = "__process=" + str(proc_id)+"_"
    str_comm_gap = "comm_gap=" + str(comm_gap)
    set_of_files = []
    for file in list:
        num_lines = sum(1 for line in open(base_path+file))
        if ((str_proc_id in file) and (str_world_size in file) and file.endswith(str(comm_gap)) and (num_lines==epochs)):
            set_of_files.append(file)
    print(set_of_files)
    # loading data to a numpy array
    my_dataz = []
    for file in set_of_files:
        my_data = genfromtxt(base_path+file, delimiter=',')
        my_dataz.append(my_data)
    avg = np.sum(my_dataz, axis=0) / len(my_dataz)
    new_file_name = dataset+"_"+"m="+str(world_size)+"_"+"c="+str(comm_gap)+"_"+"i="+str(epochs)
    np.savetxt(result_path+new_file_name, avg, delimiter=',')


# simple run
#sort_proc_comm_comp_logs(dataset="webspam",proc_id=0, world_size=32, comm_gap=1, epochs=5001)

proc_id=0
comm_gaps = [1,2,4,8,16,32,64,128,256,512,1024,2048]
all_pars = [8]
all_comm_gaps = [comm_gaps, comm_gaps, comm_gaps]
epochs=5001

for par, comm_gaps in zip(all_pars, all_comm_gaps):
    for comm_gap in comm_gaps:
        sort_proc_comm_comp_logs(dataset="webspam",proc_id=0, world_size=par, comm_gap=comm_gap, epochs=epochs)


