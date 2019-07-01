import os
import sys

base_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/2019_02_18/20/ijcnn1/ijcnn1_2019_02_20_seq_block_epochlog_5/ijcnn1/all/"
comm_gaps = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096]
comm_gaps = [1,2,4,8,512,1024]
dataset = "webspam"

list = os.listdir(base_path)
for file in list:
    for comm_gap in comm_gaps:
        str_cmg = "comm_gap="+str(comm_gap)+"_"
        if(str_cmg in file):
            print(file)
            os.rename(base_path+file,base_path+dataset+"_c="+str(comm_gap)+".csv")
