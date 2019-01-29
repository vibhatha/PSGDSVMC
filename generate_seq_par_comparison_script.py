#script generates the comparison of sequential and parallel algorithm upon the model synchronization frequency


import os
import sys


#comm_gaps = [1,2,4,8,16,32,64,128, 256, 512, 1024, 2048, 4096]
comm_gaps = [1,2,4,8,16]
comm_gaps = [32,64,128,256,512]
#comm_gaps = []

pars = [2,4,8,16,32]
mva_step=10
epoch=5001
file_suffix = "c="
for comm_gap in comm_gaps:
    file_suffix += str(comm_gap) +","
for par in pars:

    base_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/2019_01_25/"\
                +"seq_parallel_comparison_cost_acc/webspam/comparison/2/"\
                +"plot_comparison_"+str(par)+"x_"+file_suffix+".gnu"
    base_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/gnuplots/psgdsvmc/sequential_parallel_comparison/webspam/comparison/"
    base_path += "plot_comparison_"+str(par)+"x_"+file_suffix+".gnu"

    init_phrases = ["set multiplot layout "+str(len(comm_gaps))+",1 rowsfirst" + "\n",
    "set datafile separator \",\" " + "\n",
    "set autoscale x" + "\n",
    "set autoscale y" + "\n",
    "set xtics font \"Helvetica, 8\" " + "\n",
    "set ytics font \"Helvetica, 8\" " + "\n",
    "set xlabel 'Epochs' font 'Helvetica,10'" + "\n",
    "set ylabel 'Cross-Validation Accuracy' font 'Helvetica,10'" + "\n"]

    phrase = ""
    for phrase1 in init_phrases:
        phrase+= phrase1

    variable_phrase = ""

    for comm_gap in comm_gaps:
        phrase1 = "set title 'Variation of Cross-Validation Accuracy with Communication Frequency: Webspam Dataset: Parallel Model Synchronizing Frequency="+str(comm_gap)+"' font 'Helvetica,18'"
        phrase2 = "set key right bottom title 'Configuration' font 'Helvetica,10' " + "\n" + "set font 'Helvetica,8'"
        phrase3 = "plot 'webspam_p_m="+str(par)+"_c="+str(comm_gap)+"_i="+str(epoch)+"_mv="+str(mva_step)\
                  +"' using 2 title'"+str(par)+"x "+str(comm_gap)+"c "+str(epoch)+"i' with lines,"\
                  +" 'webspam_c="+str(comm_gap*par)+".csv_mv="+str(mva_step)+"' using 2 title 'Sequential "+str(comm_gap*par)+"c "+str(epoch)+"i' with lines "

        full_phrase = phrase1 + "\n" + phrase2 + "\n" + phrase3 + "\n"
        variable_phrase += full_phrase

    phrase += variable_phrase

    print(phrase)
    text_file = open(base_path, "w")
    text_file.write(phrase)
    text_file.close()