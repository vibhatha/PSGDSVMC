import sys
import os
import generate_latex_acc_n_cost_figure_script

## NOTE ##
# using 2 means acc and using 3 means the objective function value.
# the variable naming is bad it must be refactored
original_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/gnuplots/psgdsvmc/parallel_cost_cross_validation/epsilon/accuracy/fullpackage/"
base_path = "/home/vibhatha/Documents/Research/logs/psgsvmc/gnuplots/psgdsvmc/parallel_cost_cross_validation/epsilon/accuracy/fullpackage/"

def generate_epcoch_moving_average(dataset="ijcnn", epoch=5000, par=2, comms=[1,2,4,8]):
    file_suffix = "c="
    all_str = "#!/usr/bin/gnuplot\n" + "#\n\n#\n#\n" + "# AUTHOR: Vibhatha Abeykoon\n"
    all_str += "set terminal pngcairo size 1920,1080 enhanced color font 'Helvetica Bold,24'\n"
    for comm_gap in comms:
        file_suffix += str(comm_gap) +","
    png_name = "acc_n_cost_epoch_comp_"+dataset+"_"+file_suffix+"_x"+str(par)
    all_str += "set output "+"'"+png_name+".png"+"'"+"\n"
    phrase = ""
    phrase = "set multiplot layout 2,1 rowsfirst" + "\n" + \
              "set datafile separator \",\"" + "\n" + \
              "set autoscale x"+ "\n" + \
              "set autoscale y"+ "\n" + \
              "set key font 'Helvetica, 10'"+ "\n" + \
              "set xtics font \"Helvetica, 12\""+ "\n" + \
              "set ytics font \"Helvetica, 12\""+ "\n" + \
              "set xlabel 'Epochs' font 'Helvetica,14'"+ "\n" + \
              "set ylabel 'Objective Function Value' font 'Helvetica,14'"+ "\n" + \
              "set title 'Variation of Objective Function Value with Communication Frequency: Epsilon Dataset'"+ "\n" + \
              "set key right top title 'Parallel Configuration'"+ "\n" + \
              "set logscale y"+ "\n"

    phrase2 = "set ylabel 'Cross-Validation Accuracy' font 'Helvetica,14'" + "\n" + \
              "set title 'Variation of Cross-Validation Accuracy with Communication Frequency: Epsilon Dataset'" + "\n" + \
              "set key right bottom title 'Parallel Configuration'" + "\n" + "unset logscale y" + "\n"
    #"plot 'ijcnn_m=2_c=128_i=5001_mv=10' using 3 title'2x 128c 5001i' with lines,'ijcnn_m=2_c=2048_i=5001_mv=10' using 3 title'2x 2048c 5001i' with lines,'ijcnn_m=2_c=4096_i=5001_mv=10' using 3 title'2x 4096c 5001i' with lines"

    #"plot 'ijcnn_m=2_c=128_i=5001_mv=10' using 2 title'2x 128c 5001i' with lines,'ijcnn_m=2_c=2048_i=5001_mv=10' using 2 title'2x 2048c 5001i' with lines,'ijcnn_m=2_c=4096_i=5001_mv=10' using 2 title'2x 4096c 5001i' with lines"
    all_str += phrase
    s1 = "plot "
    suffix = "_mv=10"
    seq0 = ""
    seq1 = ""

    for comm_gap in comms:
        seq11 = "'"+dataset+"_m="+str(par)+"_c="+str(comm_gap)+"_i="+str(epoch)+suffix+"'"+" "
        seq12 = "using 3 title" # using 2 means acc and using 3 means the objective function value.
        seq13 = "'"+str(par)+"x "+str(comm_gap)+"c "+str(epoch)+"i"+"'" + " "
        seq14 = "with lines,"
        seq0 += (seq11 + seq12 + seq13 + seq14)

        seq21 = "'"+dataset+"_m="+str(par)+"_c="+str(comm_gap)+"_i="+str(epoch)+suffix+"'"+" "
        seq22 = "using 2 title" # using 2 means acc and using 3 means the objective function value.
        seq23 = "'"+str(par)+"x "+str(comm_gap)+"c "+str(epoch)+"i"+"'" + " "
        seq24 = "with lines,"
        seq1 += (seq21 + seq22 + seq23 + seq24)

    s11 = s1 + seq0 + "\n"
    s22 = s1 + seq1 + "\n"

    final_str =  s11 + phrase2 + s22 + "\n"
    all_str += final_str

    print(all_str)
    print("##############################################################")
    file_name = "plot_all_acc_n_cost_all_combinations_"+"_"+file_suffix+"_x"+str(par)+".gnu"
    text_file = open(original_path+file_name, "a")
    text_file.write(all_str)
    text_file.close()
    return file_name, png_name


    #s =  "plot 'ijcnn_m=2_c=128_i=5001_mv=10' using 2 title '2x 128c 5000i' with lines, 'ijcnn_m=2_c=1024_i=6001_mv=10' using 2 title '2x 1024c 6000i' with lines"


all_pars = [2,4,8,16,32]
all_comm_gaps = [[128,2048,4096], [128,2048,4096], [128, 1024, 2048], [128,512,1024],[128,256,512]]
all_comm_gaps = [[1,2,4,8], [1,2,4,8,16], [1,2,4,8,32], [1,2,4,8,128],[1,2,4,8,256], [1,2,4,8,512], [1,2,4,8,1024], [1,2,4,8,2048], [1,2,4,8,4096], [128,256,512,1024], [256,512,1024,2048,4096] ]

all_scripts = []
all_png_names = []

for par in all_pars:
    for i in range(0,len(all_comm_gaps)):
        tfile, pngname = generate_epcoch_moving_average(dataset="ijcnn", epoch=5001, par=par, comms=all_comm_gaps[i])
        all_scripts.append(tfile)
        all_png_names.append(pngname)

all_script_name = original_path+"plot_all_scripts.sh"
str1 = "#!/usr/bin/bash\n"
for item in all_scripts:
    str1 += "gnuplot "+ item + "\n"

text_file = open(all_script_name, "a")
text_file.write(str1)
text_file.close()

generate_latex_acc_n_cost_figure_script.gen_script(file_list=all_png_names)

