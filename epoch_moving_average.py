import numpy as np
import os
import sys
from numpy import genfromtxt

# for the sake of making the argument simple
# keep the different size epochs in different folders
# this is done to minimize the programming overhead in searching and re-listing files
# remedy : write a regex like pattern to sort the files by group
epochs = 6000
base_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/2019_01_12/webspam/all/results/"+str(epochs)+"/"
result_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/2019_01_12/webspam/all/results/moving_average/"+str(epochs)+"/"

epochs = 5000
base_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/2019_01_18/all/acc_cost/temp/"
result_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/2019_01_18/all/acc_cost/temp_result/"


# Reference ; https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
def moving_average(a, w=3):
    n = a.shape[0]
    d = a.shape[1]
    print(n,d)
    res = []
    range1 = np.arange(0,n-1,w) # start sequence from 0 size to n-w
    range2 = np.arange(w, n, w) # end sequeunce from window size to n
    print(len(range1), len(range2))
    for i in range(0, len(range1)):
        start = range1[i]
        end = range2[i]
        x = a[start:end]
        x_avg = np.average(x)
        res.append(x_avg)
    res = np.array(res)
    res = np.reshape(res,(len(res),1))
    return res

def moving_average_set(a, window=3) :
    n = a.shape[0]
    d = a.shape[1]
    c0 = np.reshape(a[:,0],(n,1))
    c1 = np.reshape(a[:,1],(n,1))
    c2 = np.reshape(a[:,2],(n,1))
    c3 = np.reshape(a[:,3],(n,1))
    m_c0 = moving_average(c0, window)
    m_c1 = moving_average(c1, window)
    m_c2 = moving_average(c2, window)
    m_c3 = moving_average(c3, window)
    print(m_c0.shape, m_c1.shape, m_c2.shape, m_c3.shape)
    res= np.concatenate((m_c0,m_c1,m_c2,m_c3),axis=1)
    return res


def calc_moving_average(window=10):
    list = os.listdir(base_path)
    print(list)
    for file in list:
        print(file, "In Progress")
        my_data = genfromtxt(base_path+file, delimiter=',')
        mv_my_data = moving_average_set(my_data,window=window)
        new_file_name = file + "_mv="+str(window)
        np.savetxt(result_path+new_file_name, mv_my_data, delimiter=',')
        print(file, "Completed")

calc_moving_average(10)