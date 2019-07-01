import os
import sys

base_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/2019_01_25/block_sequential/epochlogs/webspam/"
comm_gaps = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096]
dataset = "webspam"

list = os.listdir(base_path)
for file in list:
    for comm_gap in comm_gaps:
        str_cmg = "comm_gap="+str(comm_gap)+"_"
        if(str_cmg in file):
            print(file)
            os.rename(base_path+file,base_path+dataset+"_c="+str(comm_gap)+".csv")
