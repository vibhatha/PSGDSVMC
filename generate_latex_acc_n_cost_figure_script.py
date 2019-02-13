# this script can be used to generate the figure list of acc_n_cost_per_epoch for a given dataset

import os
import sys

# \begin{figure*}[htbp]
# \centering
# \includegraphics[width=1.0\textwidth,keepaspectratio]{images/parallel_execution_time/comm_comp/epsilon/all_time_32x.png}
# \caption{Distributed Training Time of Dataset Epsilon : Parallelism = 32}
# \label{fig:msf-pat-epsilon-32}
# \end{figure*}

file_path_in_sharelatex = "images/parallel_execution_time/all_time/epsilon/"
caption_prefix = "Distributed Training Time "
dataset = "Epsilon"
outpath = "/home/vibhatha/Documents/Research/logs/psgsvmc/gnuplots/psgdsvmc/parallel_cost_cross_validation/epsilon/accuracy/fullpackage/"
outfile = "epslion_acc_n_cost_figure_latex_script.tex"

def gen_script(file_list=[]):
    str1 = ""
    for file in file_list:
        str11 = ""
        str11 = "\\begin{figure*}[htbp]" + "\n"
        str11 += "\\centering" + "\n"
        str11 += "\\includegraphics[width=1.0\\textwidth,keepaspectratio]{"+file_path_in_sharelatex+file+".png}" + "\n"
        sf1 = file.split(".")[0] # acc_n_cost_epoch_comp_ijcnn_c=1,2,4,8,16,_x2.png => takes the part left to "."
        label_suffix = sf1.split("=")[1] # acc_n_cost_epoch_comp_ijcnn_c=1,2,4,8,16,_x2 => takes the part right to "="
        parallelism = label_suffix.split("_x")[1] #1,2,4,8,16,_x2 => takes the part right to x which is the parallelism
        str11 += "\\caption{"+caption_prefix+": Dataset " + dataset + " , Configuration : MSF = [" + label_suffix.split("_x")[0]  + "]"+", Parallelism = "+parallelism+"}" + "\n"
        str11 += "\\label{fig:"+"dis-msf-tr-time-"+dataset.lower()+"-x"+parallelism+"}" + "\n"
        str11 += "\\end{figure*}" + "\n"
        str11 += "\n\n"
        str1 += str11


    text_file = open(outpath+outfile, "a")
    text_file.write(str1)
    text_file.close()