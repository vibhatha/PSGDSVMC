import pandas as pd
import numpy as np
import os
import sys
import re
from matplotlib import pyplot as plt

datasource = 'covtype'
base_path = '/home/vibhatha/Documents/Research/SVM-SGD/psgd/logs/psgd_anatomy/'+datasource+'/'+datasource+'/'
result_path = '/home/vibhatha/Documents/Research/SVM-SGD/psgd/results/'+datasource+'/'
graph_path = '/home/vibhatha/Documents/Research/SVM-SGD/psgd/epoch_graphs/'

#print(list)

def generate_csv():
    list_id=0
world_sizes=[32]
world_size=1
for rank in world_sizes:
    world_size = str(rank)
    path = base_path+'/'+world_size
    list = os.listdir(path)
    list.sort()
    comms = []
    comps = []
    for file in list:
        filename = file
        df = pd.read_csv(path+"/"+file, header=None)
        world_size_string = filename[11:13]
        world_size = re.sub("[^0-9]", "", world_size_string)
        #print(world_size)
        indexes = df.iloc[:,0]
        comp = df.iloc[:,1]
        comm = df.iloc[:,2]
        comms.append(comm)
        comps.append(comp)
    comm_result = pd.concat(comms,axis=1)
    comp_result = pd.concat(comps,axis=1)
    print(comm_result)
    print("----------------------------")
    print(comp_result)
    comm_result.to_csv(result_path+"comms_"+str(rank)+"_.csv")
    comp_result.to_csv(result_path+"comps_"+str(rank)+"_.csv")

def gen_graphs():
    list = os.listdir(result_path)
    list.sort()
    for file in list:
        name = file.split(".")[0]
        type = name.split("_")[0]
        procs = name.split("_")[1]
        df = pd.read_csv(result_path+"/"+file, header=None)
        indexes = df.iloc[:,0]
        data_cols = len(df.columns)
        items = []
        for i in range(1, data_cols):
            items.append(df.iloc[:,i])
            plt.plot(indexes,df.iloc[:,i], label='process_'+str(i))
        plt.legend(loc='upper right')
        if(type=='comms'):
            plt.title("Per Process Communication Time Variation against Epochs", fontsize=16, fontweight='bold')
        if(type=='comps'):
            plt.title("Per Process Computation Time Variation against Epochs", fontsize=16, fontweight='bold')
        plt.suptitle(datasource, fontsize=10)
        plt.xlabel("Epochs")
        plt.ylabel("Time (s)")
        my_dpi = 200
        plt.savefig(graph_path+type+'_'+str(procs)+".png",dpi=my_dpi)
        plt.show()

gen_graphs()