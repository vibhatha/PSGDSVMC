import os
import sys

base_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/2019_01_25/parallel/epsilon/epochlog/comms_comp/epsilon/"

list = os.listdir(base_path)
for file in list:
    num_lines = sum(1 for line in open(base_path+file))
    if(num_lines==5001 and "world_size=8_" in file):
        print(file)