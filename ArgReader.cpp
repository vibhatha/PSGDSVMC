//
// Created by vibhatha on 11/7/18.
//

#include <sstream>
#include "ArgReader.h"

ArgReader::ArgReader(int argc, char **argv) : argc(argc), argv(argv) {

}

OptArgs ArgReader::getParams() {
    OptArgs optArgs;
//    cout << "You have entered " << argc
//         << " arguments:" << "\n";

    for (int i = 1; i < argc; ++i){
        string comp = "-dataset";
        if(comp.compare(argv[i])==0){
            optArgs.setDataset(argv[i+1]);
        }
        comp = "-itr";
        if(comp.compare(argv[i])==0){
            stringstream geek(argv[i+1]);
            int x = 0;
            geek >> x;
            optArgs.setIterations(x);
        }
        comp = "-alpha";
        if(comp.compare(argv[i])==0){
            stringstream geek(argv[i+1]);
            double x = 0;
            geek >> x;
            optArgs.setAlpha(x);
        }
        comp = "-features";
        if(comp.compare(argv[i])==0){
            stringstream geek(argv[i+1]);
            int x = 0;
            geek >> x;
            optArgs.setFeatures(x);
        }
        comp = "-trainingSamples";
        if(comp.compare(argv[i])==0){
            stringstream geek(argv[i+1]);
            int x = 0;
            geek >> x;
            optArgs.setTrainingSamples(x);
        }
        comp = "-testingSamples";
        if(comp.compare(argv[i])==0){
            stringstream geek(argv[i+1]);
            int x = 0;
            geek >> x;
            optArgs.setTestingSamples(x);
        }

        comp = "-split";
        if(comp.compare(argv[i])==0){
            optArgs.setIsSplit(true);
        }

        comp = "-ratio";
        if(comp.compare(argv[i])==0){
            stringstream geek(argv[i+1]);
            double x = 0;
            geek >> x;
            optArgs.setRatio(x);
        }

        comp = "-threads";
        if(comp.compare(argv[i])==0){
            stringstream geek(argv[i+1]);
            int x = 0;
            geek >> x;
            optArgs.setThreads(x);
        }

        comp = "-workers";
        if(comp.compare(argv[i])==0){
            stringstream geek(argv[i+1]);
            int x = 0;
            geek >> x;
            optArgs.setWorkers(x);
        }

        comp = "-et";
        if(comp.compare(argv[i])==0){
            optArgs.setIsEpochTime(true);
        }

        comp = "-nt";
        if(comp.compare(argv[i])==0){
            optArgs.setIsNormalTime(true);
        }

        comp = "-bulk";
        if(comp.compare(argv[i])==0){
            optArgs.setBulk(true);
        }

        comp = "-batch";
        if(comp.compare(argv[i])==0){
            optArgs.setBatch(true);
        }

        comp = "-batch_size";
        if(comp.compare(argv[i])==0){
            stringstream geek(argv[i+1]);
            double x = 0;
            geek >> x;
            optArgs.setBatch_per(x);
        }

        comp = "-drop";
        if(comp.compare(argv[i])==0){
            optArgs.setIsDrop(true);
        }

        comp = "-drop_per";
        if(comp.compare(argv[i])==0){
            stringstream geek(argv[i+1]);
            double x = 0;
            geek >> x;
            optArgs.setDrop_out_per(x);
        }

        comp = "-seq";
        if(comp.compare(argv[i])==0){
            optArgs.setSequential(true);
        }

        comp = "-ring";
        if(comp.compare(argv[i])==0){
            optArgs.setRing(true);
        }

        comp = "-randomringv1";
        if(comp.compare(argv[i])==0){
            optArgs.setRandomringv1(true);
        }

        comp = "-randomringv2";
        if(comp.compare(argv[i])==0){
            optArgs.setRandomringv2(true);
        }

        comp = "-fullbatch";
        if(comp.compare(argv[i])==0){
            optArgs.setFullbatchv1(true);
        }

        comp = "-pegasos";
        if(comp.compare(argv[i])==0){
            optArgs.setPegasos(true);
        }

        comp = "-pegasosfb";
        if(comp.compare(argv[i])==0){
            optArgs.setPegasosFullBatch(true);
        }

        comp = "-pegasosb";
        if(comp.compare(argv[i])==0){
            optArgs.setPegasosBatch(true);
        }

        comp = "-pegasosbt";
        if(comp.compare(argv[i])==0){
            optArgs.setPegasosBatchThreaded(true);
        }

        comp = "-error";
        if(comp.compare(argv[i])==0){
            stringstream geek(argv[i+1]);
            double x = 0;
            geek >> x;
            optArgs.setError_threshold(x);
        }

        comp = "-threads";
        if(comp.compare(argv[i])==0){
            stringstream geek(argv[i+1]);
            int x = 0;
            geek >> x;
            optArgs.setThreads(x);
        }

        comp = "-pegasosbs";
        if(comp.compare(argv[i])==0){
            optArgs.setPegasosBlockSequential(true);
        }

        comp = "-notime";
        if(comp.compare(argv[i])==0){
            optArgs.setPegasosSeqNoTime(true);
        }


        //cout  << i  << ", " << argv[i] << "\n";
    }

    return optArgs;
}