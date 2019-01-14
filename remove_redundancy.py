import numpy as np
import os
import sys
from numpy import genfromtxt

base_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/2019_01_11/ijcnn1/"
result_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/2019_01_11/results/group/"

def remove_redundancy(epoch=5000,dataset='ijcnn'):
    world_sizes=[32]
    list = os.listdir(base_path)
    #print(list)

    for world_size in world_sizes:
        world_file_set = []
        for file in list:
            world_str = "world_size_" + str(world_size) + "_"
            if (world_str in file):
                world_file_set.append(file)
        #print(world_file_set)
        comm_gaps = ['1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024', '2048', '4096']
        comm_gaps = ['1024']
        sorted_list = []
        # sort files in the order of the communication gap
        for comm_gap in comm_gaps:
            comm_gap = 'comm_gap=' + comm_gap + '_'
            for file in world_file_set:
                if(comm_gap in file):
                    sorted_list.append(file)
        groups_by_commgap = []
        for comm_gap in comm_gaps:
            group_by_commgap = []
            for file in sorted_list:
                str_comm_gap = 'comm_gap=' + comm_gap + '_'
                if(str_comm_gap in file):
                    num_lines = sum(1 for line in open(base_path+file))
                    #print(num_lines, file)
                    if(num_lines==epoch):
                        group_by_commgap.append(file)
            groups_by_commgap.append(group_by_commgap)
        counter=0
        for grp in groups_by_commgap:
            print("-------------------------------")
            all_val = []
            for file in grp:
                num_lines = sum(1 for line in open(base_path+file))
                print(num_lines,file)
                my_data = genfromtxt(base_path+file, delimiter=',')
                all_val.append(my_data)
            all_val = np.array(all_val)
            avg_all_val = np.average(all_val,axis=0)
            save_file_name = result_path+dataset+"_m="+str(world_size)+"_c="+str(comm_gaps[counter])+"_i="+str(epoch)
            np.savetxt(save_file_name, avg_all_val, delimiter=',')
            counter = counter + 1

            print("-------------------------------")
remove_redundancy(6001,'ijcnn')