import sys
import os


def generate_epcoch_moving_average(dataset="ijcnn", epoch=5000, pars=[], comms=[]):
    s1 = "plot "
    suffix = "_mv=10"
    seq = ""
    for par in pars:
        for comm_gap in comm_gaps:
            seq1 = "'"+dataset+"_m="+str(par)+"_c="+str(comm_gap)+"_i="+str(epoch)+suffix+"'"+" "
            seq2 = "using 2 title"
            seq3 = "'"+str(par)+"x "+str(comm_gap)+"c "+str(epoch)+"i"+"'" + " "
            seq4 = "with lines,"
            seq += (seq1 + seq2 + seq3 + seq4)

    #s =  "plot 'ijcnn_m=2_c=128_i=5001_mv=10' using 2 title '2x 128c 5000i' with lines, 'ijcnn_m=2_c=1024_i=6001_mv=10' using 2 title '2x 1024c 6000i' with lines"
    s = s1 + seq
    print(s)

pars = [2,4]
comm_gaps = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096]
generate_epcoch_moving_average(dataset="ijcnn", epoch=5001, pars=pars, comms=comm_gaps)