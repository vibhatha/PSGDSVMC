#include <iostream>
#include "Test.h"
#include "DataSet.h"
#include "Util.h"
#include "SGD.h"
#include "ArgReader.h"
#include "Predict.h"
#include "ResourceManager.h"
#include "PSGD.h"
#include "Initializer.h"
#include <ctime>
#include <mpi.h>
#include <ctime>
#include <fstream>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>


using namespace std;

void test1();

void test2();

void test3();

void test4();

void test5();

void test6();

void test7();

void test8(int argc, char **argv);

void sgd();

void train(OptArgs optArgs);

void parallel(OptArgs optArgs);

void parallelLoad(OptArgs optArgs);

void parallelLoadBatchV1(OptArgs optArgs);

void parallelLoadBatchV2(OptArgs optArgs, int comm_gap);

void parallelLoadRotationV1(OptArgs optArgs);

void parallelLoadRandomV1(OptArgs optArgs);

void parallelLoadRandomV2(OptArgs optArgs);

void trainSequential(OptArgs optArgs);

void sequentialLoad(OptArgs optArgs);

void parallelFullBatchv1(OptArgs optArgs);

void sequentialPegasos(OptArgs optArgs);

void parallelPegasosFullBatchV1(OptArgs optArgs);

void parallelPegasosBatchV1(OptArgs optArgs, int comm_gap);

void sequentialPegasosBatchV1(OptArgs optArgs, int comm_gap);

void mpiTest();

int getdir (string dir, vector<string> &files);

string getTimeStamp();

void summary(string logfile, int world_size, double acc, double time, string datasource);

void summary(string logfile, int world_size, double acc, double time, string datasource, double alpha);

void summary(string logfile, int world_size, double acc, double time, string datasource, double alpha, double error_threshold);

int main(int argc, char **argv) {
    //std::cout << "Hello, World!" << std::endl;

    ArgReader argReader(argc, argv);
    OptArgs optArgs = argReader.getParams();
    //train(optArgs);
    //sgd();
    //test4();
    //test5();
    //parallelLoad(optArgs);
    //parallelLoadBatchV1(optArgs);
    //trainSequential(optArgs);
    //sequentialLoad(optArgs);

//    if(optArgs.isBatch()) {
//        double per = optArgs.getBatch_per();
//        int sample_gap = per * optArgs.getTrainingSamples();
//        parallelLoadBatchV2(optArgs, sample_gap);
//    }else {
//        parallelLoadBatchV1(optArgs);
//    }
    //test8(argc,argv);

    //test7();
    //mpiTest();
    if(optArgs.isSequential()) {
        trainSequential(optArgs);
    } else if(optArgs.isRing()){
        parallelLoadRotationV1(optArgs);
    } else if(optArgs.isRandomringv1()) {
        parallelLoadRandomV1(optArgs);
    } else if(optArgs.isRandomringv2()) {
        parallelLoadRandomV2(optArgs);
    } else if(optArgs.isBatch()) {
        double per = optArgs.getBatch_per();
        int sample_gap = per * optArgs.getTrainingSamples();
        parallelLoadBatchV2(optArgs, sample_gap);
    } else if(optArgs.isFullbatchv1()) {
        parallelFullBatchv1(optArgs);
    } else if(optArgs.isPegasos()) {
        sequentialPegasos(optArgs);
    } else if(optArgs.isPegasosFullBatch()) {
        parallelPegasosFullBatchV1(optArgs);
    } else if(optArgs.isPegasosBatch()) {
        double per = optArgs.getBatch_per();
        parallelPegasosBatchV1(optArgs, per);
    } else if(optArgs.isPegasosBlockSequential()) {
        sequentialPegasosBatchV1(optArgs, optArgs.getBatch_per());
    }

    return 0;
}

void mpiTest() {
    MPI_Init(NULL, NULL);
//    int world_rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//    int world_size;
//    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//    cout << "World Rank : " << world_rank << ", World Size : " << world_size << endl;
    MPI_Finalize();
}

void parallelPegasosFullBatchV1(OptArgs optArgs) {
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //optArgs.toString();
    ResourceManager resourceManager;
    Initializer initializer;
    Util util;
    resourceManager.loadDataSourcePath();
    resourceManager.loadLogSourcePath();
    resourceManager.loadSummaryPath();
    resourceManager.loadWeightSummaryPath();
    resourceManager.loadEpochSummaryPath();
    resourceManager.loadCommCompSummaryPath();
    string commcomplogfile="";
    commcomplogfile.append(resourceManager.getCommcompSummaryBasePath()).append("pegasos/fullbatch/").append(optArgs.getDataset()).append("/").append("comm_comp_time_").append(getTimeStamp()).append("_alpha_").append(to_string(optArgs.getAlpha())).append(".csv");
    string summarylogfile ="";
    summarylogfile.append(resourceManager.getLogSummaryBasePath()).append("/parallel/").append("pegasos/fullbatch/").append(optArgs.getDataset()).append("/").append("summary_comm_gap=").append(to_string(optArgs.getBatch_per())).append(".csv");
    string weightlogfile = "";
    weightlogfile.append(resourceManager.getWeightSummaryBasePath()).append("/").append(optArgs.getDataset()).append("/").append(getTimeStamp())
            .append("_").append("_alpha_").append(to_string(optArgs.getAlpha())).append("pegasos_full_batch_weight_summary.csv");
    string epochlogfile = resourceManager.getEpochlogSummaryBasePath();
    epochlogfile.append("parallel/pegasos/fullbatch/").append(optArgs.getDataset()).append("/").append(getTimeStamp()).append("_world_size_").append(to_string(world_size)).append("_rank_").append(to_string(world_rank)).append("_").append("alpha_").append(to_string(optArgs.getAlpha())).append("_pegasos_fullbatch_cross_validation_accuracy.csv");
    string logfile = "";
    if (optArgs.isIsSplit()) {
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string logsourceBase = resourceManager.getLogSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string sourceFile;
        sourceFile.append(datasourceBase).append(datasource).append(trainFileName);
        cout << "SourceFile : " << sourceFile << endl;
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples * ratio;
        int testSet = totalSamples - trainSet;
        int dataPerMachine = trainSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;

        double *w = new double[features];

        double ytrain[dataPerMachine];
        initializer.initializeWeightsWithArray(dataPerMachine, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[dataPerMachine];
        for (int i = 0; i < dataPerMachine; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        DataSet dataSet(features, trainingSamples, testingSamples, optArgs.isIsSplit(), optArgs.getRatio(), sourceFile,
                        world_size, world_rank);
        dataSet.distributedLoad(Xtrain, ytrain, Xtest, ytest);
//        dataPerMachine = dataSet.getDataPerMachine();
//        if(world_rank==0) {
//            cout << "From Main : " << "Data Per Machine : " << dataPerMachine << endl;
//            Util util;
//            util.print2DMatrix(Xtest, 20, features);
//            printf("\n----------------------------------------\n");
//        }


        PSGD sgd1(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, dataPerMachine,
                  testingSamples, world_size, world_rank, Xtest, ytest);
        sgd1.setError_threshold(optArgs.getError_threshold());
        double startTime = MPI_Wtime();
        if (optArgs.isIsNormalTime()) {//works with -nt flag
            sgd1.pegasosSGDFullBatchv1(w, epochlogfile);
        }

        if (optArgs.isIsEpochTime()) {
            //TODO: this method must be written
            //sgd1.pegasosSGDFullBatchv1(w, epochlogfile);
        }

        double endTime = MPI_Wtime();
        if (world_rank == 0) {
            double totalTrainingTime = ((endTime - startTime)-sgd1.getTotalPredictionTime());
            cout << "Training Time : " << totalTrainingTime << endl;
            Predict predict(Xtest, ytest, w, testSet, features);
            double acc = predict.predict();
            cout << "Testing Accuracy : " << acc << "%" << endl;
            summary(summarylogfile, world_size, acc, totalTrainingTime, datasource, optArgs.getAlpha());
            util.writeWeight(w, features, weightlogfile);
        }

        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;
    } else {
        if(world_rank==0) {
            cout << "Training with full data set..." << endl;
        }
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string logsourceBase = resourceManager.getLogSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string trainFile;
        trainFile.append(datasourceBase).append(datasource).append(trainFileName);
        string testFile;
        testFile.append(datasourceBase).append(datasource).append(testFileName);
        cout << "Train File : " << trainFile << endl;
        cout << "Test File : " << testFile << endl;
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples;
        int testSet = testingSamples;
        int dataPerMachine = trainSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;

        double *w = new double[features];

        double ytrain[dataPerMachine];
        initializer.initializeWeightsWithArray(dataPerMachine, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[dataPerMachine];
        for (int i = 0; i < dataPerMachine; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        DataSet dataSet(features, trainingSamples, testingSamples, trainFile, testFile, world_size, world_rank, optArgs.isBulk());
        dataSet.distributedLoad(Xtrain, ytrain, Xtest, ytest);
//        dataPerMachine = dataSet.getDataPerMachine();
//        if(world_rank==0) {
//            cout << "From Main : " << "Data Per Machine : " << dataPerMachine << endl;
//            Util util;
//            util.print2DMatrix(Xtest, 20, features);
//            printf("\n----------------------------------------\n");
//        }


        PSGD sgd1(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, dataPerMachine,
                  testingSamples, world_size, world_rank);
        double startTime = MPI_Wtime();

        if (optArgs.isIsNormalTime()) {
            sgd1.pegasosSGDFullBatchv1(w, epochlogfile);
        }

        if (optArgs.isIsEpochTime()) {
            //TODO this method must be written
        }

        double endTime = MPI_Wtime();
        if (world_rank == 0) {

            double acc = 0;
            if(!optArgs.isBulk()){
                Predict predict(Xtest, ytest, w, testSet, features);
                acc = predict.predict();
            }else{
                string bulktestfile;
                bulktestfile.append(datasourceBase).append(datasource).append("/bulk/");
                string dir = string(bulktestfile);
                vector<string> files = vector<string>();

                getdir(dir,files);
                int fixedTest = 20000;
                double ytest[fixedTest];
                initializer.initializeWeightsWithArray(testSet, ytest);

                double **Xtest;

                Xtest = new double *[fixedTest];
                for (int i = 0; i < fixedTest; ++i) {
                    Xtest[i] = new double[features];
                }
                double cum_acc = 0;
                for (unsigned int i = 0;i < files.size()-1;i++) {
                    string file="";
                    file.append(bulktestfile).append(files[i]);
                    cout << file << endl;
                    DataSet dataSet1(features, fixedTest,file);
                    dataSet1.loadTestData(Xtest, ytest);
                    Predict predict(Xtest, ytest, w, testSet, features);
                    acc = predict.predict();
                    cout << "Test " << i << ", Accuracy : " << acc << endl;
                    cum_acc += acc;
                }
                acc = cum_acc / double(files.size());
            }
            double totalTrainingTime = ((endTime - startTime) - sgd1.getTotalPredictionTime());
            cout << "Training Time : " << totalTrainingTime << endl;
            cout << "Testing Accuracy : " << acc << "%" << endl;
            summary(summarylogfile, world_size, acc, totalTrainingTime, datasource, optArgs.getAlpha());
            util.writeWeight(w, features, weightlogfile);
        }

        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;
    }
    MPI_Finalize();
}

void parallelPegasosBatchV1(OptArgs optArgs, int comm_gap) {
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //optArgs.toString();
    ResourceManager resourceManager;
    Initializer initializer;
    Util util;
    resourceManager.loadDataSourcePath();
    resourceManager.loadLogSourcePath();
    resourceManager.loadSummaryPath();
    resourceManager.loadWeightSummaryPath();
    resourceManager.loadEpochSummaryPath();
    resourceManager.loadCommCompSummaryPath();
    string summarylogfile ="";
    summarylogfile.append(resourceManager.getLogSummaryBasePath()).append("/parallel/pegasos/").append("/batch/").append(optArgs.getDataset()).append("/").append("summary_comm_gap=").append(to_string(optArgs.getBatch_per())).append(".csv");
    string weightlogfile = "";
    string epochweightlogfile = "";
    weightlogfile.append(resourceManager.getWeightSummaryBasePath()).append("/").append(optArgs.getDataset()).append("/").append(getTimeStamp())
            .append("_").append("_alpha_").append(to_string(optArgs.getAlpha())).append("batch_weight_summary.csv");
    epochweightlogfile.append(resourceManager.getWeightSummaryBasePath()).append("/parallel/pegasos/batch/").append(optArgs.getDataset()).append("/").append(getTimeStamp()).append("alpha_").append(to_string(optArgs.getAlpha())).append("_comm_gap=").append(to_string(comm_gap)).append("_epoch_weightlog.csv");
    string epochlogfile = resourceManager.getEpochlogSummaryBasePath();
    epochlogfile.append("parallel/pegasos/batch/").append(optArgs.getDataset()).append("/").append(getTimeStamp()).append("_world_size_").append(to_string(world_size)).append("_rank_").append(to_string(world_rank)).append("_alpha_").append(to_string(optArgs.getAlpha())).append("_batch_cross_validation_accuracy.csv");
    string commcomplogfile = "";
    commcomplogfile.append(resourceManager.getCommcompSummaryBasePath()).append("parallel/pegasos/batch/").append(optArgs.getDataset()).append("/");
    if (optArgs.isIsSplit()) {
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string logsourceBase = resourceManager.getLogSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string sourceFile;
        sourceFile.append(datasourceBase).append(datasource).append(trainFileName);
        cout << "SourceFile : " << sourceFile << endl;
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples * ratio;
        int testSet = totalSamples - trainSet;
        int dataPerMachine = trainSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;

        if(world_rank==0) {
            cout << "Comm Gap : " << comm_gap << endl;
        }
        double *w = new double[features];

        double ytrain[dataPerMachine];
        initializer.initializeWeightsWithArray(dataPerMachine, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[dataPerMachine];
        for (int i = 0; i < dataPerMachine; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        DataSet dataSet(features, trainingSamples, testingSamples, optArgs.isIsSplit(), optArgs.getRatio(), sourceFile,
                        world_size, world_rank);
        dataSet.distributedLoad(Xtrain, ytrain, Xtest, ytest);
//        dataPerMachine = dataSet.getDataPerMachine();
//        if(world_rank==0) {
//            cout << "From Main : " << "Data Per Machine : " << dataPerMachine << endl;
//            Util util;
//            util.print2DMatrix(Xtest, 20, features);
//            printf("\n----------------------------------------\n");
//        }


        PSGD sgd1(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, dataPerMachine,
                  testingSamples, world_size, world_rank, Xtest, ytest);
        sgd1.setError_threshold(optArgs.getError_threshold());
        double startTime = MPI_Wtime();
        if (optArgs.isIsNormalTime()) {
            sgd1.pegasosSGDBatchv2(w, comm_gap, commcomplogfile, epochlogfile, epochweightlogfile);
        }

        if (optArgs.isIsEpochTime()) {
            //TODO this method must be written in PSGD end
        }


        double endTime = MPI_Wtime();
        if (world_rank == 0) {
            double trainingTime = (endTime - startTime) - (sgd1.getTotalPredictionTime());
            cout << "Training Time : " << trainingTime << endl;
            Predict predict(Xtest, ytest, w, testSet, features);
            double acc = predict.predict();
            cout << "Testing Accuracy : " << acc << "%" << endl;
            summary(summarylogfile, world_size, acc, trainingTime, datasource, optArgs.getAlpha(), sgd1.getError_threshold());
            util.writeWeight(w, features, weightlogfile);
        }

        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;
    } else {
        if(world_rank==0) {
            cout << "Training with full data set..." << endl;
        }
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string logsourceBase = resourceManager.getLogSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string trainFile;
        trainFile.append(datasourceBase).append(datasource).append(trainFileName);
        string testFile;
        testFile.append(datasourceBase).append(datasource).append(testFileName);
        cout << "Train File : " << trainFile << endl;
        cout << "Test File : " << testFile << endl;
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples;
        int testSet = testingSamples;
        int dataPerMachine = trainSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;
        comm_gap = optArgs.getBatch_per() * dataPerMachine;
        if(world_rank==0) {
            cout << "Comm Gap : " << comm_gap << endl;
        }
        double *w = new double[features];

        double ytrain[dataPerMachine];
        initializer.initializeWeightsWithArray(dataPerMachine, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[dataPerMachine];
        for (int i = 0; i < dataPerMachine; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        DataSet dataSet(features, trainingSamples, testingSamples, trainFile, testFile, world_size, world_rank, optArgs.isBulk());
        dataSet.distributedLoad(Xtrain, ytrain, Xtest, ytest);
//        dataPerMachine = dataSet.getDataPerMachine();
//        if(world_rank==0) {
//            cout << "From Main : " << "Data Per Machine : " << dataPerMachine << endl;
//            Util util;
//            util.print2DMatrix(Xtest, 20, features);
//            printf("\n----------------------------------------\n");
//        }


        PSGD sgd1(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, dataPerMachine,
                  testingSamples, world_size, world_rank);
        double startTime = MPI_Wtime();
        if (optArgs.isIsNormalTime()) {
            sgd1.pegasosSGDBatchv2(w, comm_gap, commcomplogfile, epochlogfile, epochweightlogfile);
            //sgd1.adamSGDBatchv2(w, comm_gap);
        }

        if (optArgs.isIsEpochTime()) {
            //TODO this section must be written
        }


        double endTime = MPI_Wtime();
        if (world_rank == 0) {

            double acc = 0;
            if(!optArgs.isBulk()){
                Predict predict(Xtest, ytest, w, testSet, features);
                acc = predict.predict();
            }else{
                string bulktestfile;
                bulktestfile.append(datasourceBase).append(datasource).append("/bulk/");
                string dir = string(bulktestfile);
                vector<string> files = vector<string>();

                getdir(dir,files);
                int fixedTest = 20000;
                double ytest[fixedTest];
                initializer.initializeWeightsWithArray(testSet, ytest);

                double **Xtest;

                Xtest = new double *[fixedTest];
                for (int i = 0; i < fixedTest; ++i) {
                    Xtest[i] = new double[features];
                }
                double cum_acc = 0;
                for (unsigned int i = 0;i < files.size()-1;i++) {
                    string file="";
                    file.append(bulktestfile).append(files[i]);
                    cout << file << endl;
                    DataSet dataSet1(features, fixedTest,file);
                    dataSet1.loadTestData(Xtest, ytest);
                    Predict predict(Xtest, ytest, w, testSet, features);
                    acc = predict.predict();
                    cout << "Test " << i << ", Accuracy : " << acc << endl;
                    cum_acc += acc;
                }
                acc = cum_acc / double(files.size());
            }
            cout << "Training Time : " << (endTime - startTime) << endl;
            cout << "Testing Accuracy : " << acc << "%" << endl;
            summary(summarylogfile, world_size, acc, (endTime - startTime), datasource, optArgs.getAlpha());
            util.writeWeight(w, features, weightlogfile);
        }

        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;
    }
    MPI_Finalize();
}

void parallelFullBatchv1(OptArgs optArgs) {
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //optArgs.toString();
    ResourceManager resourceManager;
    Initializer initializer;
    Util util;
    resourceManager.loadDataSourcePath();
    resourceManager.loadLogSourcePath();
    resourceManager.loadSummaryPath();
    resourceManager.loadWeightSummaryPath();
    resourceManager.loadEpochSummaryPath();
    resourceManager.loadCommCompSummaryPath();
    string exptype = "adam";
    string commcomplogfile="";
    commcomplogfile.append(resourceManager.getCommcompSummaryBasePath()).append("fullbatch/").append(optArgs.getDataset()).append("/").append("comm_comp_time_").append(getTimeStamp()).append(".csv");
    string summarylogfile ="";
    summarylogfile.append(resourceManager.getLogSummaryBasePath()).append("/parallel/").append(exptype).append("/fullbatch/").append(optArgs.getDataset()).append("/").append("summary_comm_gap=").append(to_string(optArgs.getBatch_per())).append(".csv");
    string weightlogfile = "";
    weightlogfile.append(resourceManager.getWeightSummaryBasePath()).append("/").append(optArgs.getDataset()).append("/").append(getTimeStamp())
            .append("_").append("full_batch_weight_summary.csv");
    string epochlogfile = resourceManager.getEpochlogSummaryBasePath();
    epochlogfile.append("parallel/").append(exptype).append("/fullbatch/").append(optArgs.getDataset()).append("/").append(getTimeStamp()).append("_").append("rank_").append(to_string(world_rank)).append("_").append("full_batch_cross_validation_accuracy.csv");
    string logfile = "";
    if (optArgs.isIsSplit()) {
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string logsourceBase = resourceManager.getLogSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string sourceFile;
        sourceFile.append(datasourceBase).append(datasource).append(trainFileName);
        cout << "SourceFile : " << sourceFile << endl;
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples * ratio;
        int testSet = totalSamples - trainSet;
        int dataPerMachine = trainSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;

        double *w = new double[features];

        double ytrain[dataPerMachine];
        initializer.initializeWeightsWithArray(dataPerMachine, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[dataPerMachine];
        for (int i = 0; i < dataPerMachine; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        DataSet dataSet(features, trainingSamples, testingSamples, optArgs.isIsSplit(), optArgs.getRatio(), sourceFile,
                        world_size, world_rank);
        dataSet.distributedLoad(Xtrain, ytrain, Xtest, ytest);
//        dataPerMachine = dataSet.getDataPerMachine();
//        if(world_rank==0) {
//            cout << "From Main : " << "Data Per Machine : " << dataPerMachine << endl;
//            Util util;
//            util.print2DMatrix(Xtest, 20, features);
//            printf("\n----------------------------------------\n");
//        }


        PSGD sgd1(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, dataPerMachine,
                  testingSamples, world_size, world_rank, Xtest, ytest);
        double startTime = MPI_Wtime();
        if (optArgs.isIsNormalTime()) {
            sgd1.sgdFullBatchv2(w, epochlogfile);
        }

        if (optArgs.isIsEpochTime()) {
            sgd1.sgdFullBatchv1(w, commcomplogfile, epochlogfile);
            string suffix = getTimeStamp();
            logfile.append(logsourceBase).append("logs/epochlog/").append(datasource).append("/").append(
                    "world_size=").append(to_string(world_size)).append("_iterations=").append(
                    to_string(optArgs.getIterations()));
            logfile.append("_").append(suffix);
        }

        double endTime = MPI_Wtime();
        if (world_rank == 0) {
            double totalTrainingTime = ((endTime - startTime)-sgd1.getTotalPredictionTime());
            cout << "Training Time : " << totalTrainingTime << endl;
            Predict predict(Xtest, ytest, w, testSet, features);
            double acc = predict.predict();
            cout << "Testing Accuracy : " << acc << "%" << endl;
            summary(summarylogfile, world_size, acc, totalTrainingTime, datasource);
            util.writeWeight(w, features, weightlogfile);
        }

        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;
    } else {
        if(world_rank==0) {
            cout << "Training with full data set..." << endl;
        }
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string logsourceBase = resourceManager.getLogSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string trainFile;
        trainFile.append(datasourceBase).append(datasource).append(trainFileName);
        string testFile;
        testFile.append(datasourceBase).append(datasource).append(testFileName);
        cout << "Train File : " << trainFile << endl;
        cout << "Test File : " << testFile << endl;
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples;
        int testSet = testingSamples;
        int dataPerMachine = trainSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;

        double *w = new double[features];

        double ytrain[dataPerMachine];
        initializer.initializeWeightsWithArray(dataPerMachine, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[dataPerMachine];
        for (int i = 0; i < dataPerMachine; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        DataSet dataSet(features, trainingSamples, testingSamples, trainFile, testFile, world_size, world_rank, optArgs.isBulk());
        dataSet.distributedLoad(Xtrain, ytrain, Xtest, ytest);
//        dataPerMachine = dataSet.getDataPerMachine();
//        if(world_rank==0) {
//            cout << "From Main : " << "Data Per Machine : " << dataPerMachine << endl;
//            Util util;
//            util.print2DMatrix(Xtest, 20, features);
//            printf("\n----------------------------------------\n");
//        }


        PSGD sgd1(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, dataPerMachine,
                  testingSamples, world_size, world_rank);
        double startTime = MPI_Wtime();
        if (optArgs.isIsNormalTime()) {
            sgd1.sgdFullBatchv2(w, epochlogfile);
        }

        if (optArgs.isIsEpochTime()) {
            string suffix = getTimeStamp();
            logfile.append(logsourceBase).append("logs/epochlog/").append(datasource).append("/").append(
                    "world_size=").append(to_string(world_size)).append("_iterations=").append(
                    to_string(optArgs.getIterations()));
            logfile.append("_").append(suffix);
            sgd1.sgdFullBatchv1(w, commcomplogfile, epochlogfile);
        }


        double endTime = MPI_Wtime();
        if (world_rank == 0) {

            double acc = 0;
            if(!optArgs.isBulk()){
                Predict predict(Xtest, ytest, w, testSet, features);
                acc = predict.predict();
            }else{
                string bulktestfile;
                bulktestfile.append(datasourceBase).append(datasource).append("/bulk/");
                string dir = string(bulktestfile);
                vector<string> files = vector<string>();

                getdir(dir,files);
                int fixedTest = 20000;
                double ytest[fixedTest];
                initializer.initializeWeightsWithArray(testSet, ytest);

                double **Xtest;

                Xtest = new double *[fixedTest];
                for (int i = 0; i < fixedTest; ++i) {
                    Xtest[i] = new double[features];
                }
                double cum_acc = 0;
                for (unsigned int i = 0;i < files.size()-1;i++) {
                    string file="";
                    file.append(bulktestfile).append(files[i]);
                    cout << file << endl;
                    DataSet dataSet1(features, fixedTest,file);
                    dataSet1.loadTestData(Xtest, ytest);
                    Predict predict(Xtest, ytest, w, testSet, features);
                    acc = predict.predict();
                    cout << "Test " << i << ", Accuracy : " << acc << endl;
                    cum_acc += acc;
                }
                acc = cum_acc / double(files.size());
            }
            double totalTrainingTime = ((endTime - startTime) - sgd1.getTotalPredictionTime());
            cout << "Training Time : " << totalTrainingTime << endl;
            cout << "Testing Accuracy : " << acc << "%" << endl;
            summary(summarylogfile, world_size, acc, totalTrainingTime, datasource);
            util.writeWeight(w, features, weightlogfile);
        }

        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;
    }
    MPI_Finalize();
}



void parallelLoad(OptArgs optArgs) {
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    optArgs.toString();
    ResourceManager resourceManager;
    Initializer initializer;
    resourceManager.loadDataSourcePath();
    resourceManager.loadLogSourcePath();
    resourceManager.loadSummaryPath();
    string summarylogfile ="";
    summarylogfile.append(resourceManager.getLogSummaryBasePath()).append("/").append(optArgs.getDataset()).append("/summary.csv");
    string logfile = "";
    if (optArgs.isIsSplit()) {
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string logsourceBase = resourceManager.getLogSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string sourceFile;
        sourceFile.append(datasourceBase).append(datasource).append(trainFileName);
        cout << "SourceFile : " << sourceFile << endl;
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples * ratio;
        int testSet = totalSamples - trainSet;
        int dataPerMachine = trainSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;

        double *w = new double[features];

        double ytrain[dataPerMachine];
        initializer.initializeWeightsWithArray(dataPerMachine, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[dataPerMachine];
        for (int i = 0; i < dataPerMachine; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        DataSet dataSet(features, trainingSamples, testingSamples, optArgs.isIsSplit(), optArgs.getRatio(), sourceFile,
                        world_size, world_rank);
        dataSet.distributedLoad(Xtrain, ytrain, Xtest, ytest);
//        dataPerMachine = dataSet.getDataPerMachine();
//        if(world_rank==0) {
//            cout << "From Main : " << "Data Per Machine : " << dataPerMachine << endl;
//            Util util;
//            util.print2DMatrix(Xtest, 20, features);
//            printf("\n----------------------------------------\n");
//        }


        PSGD sgd1(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, dataPerMachine,
                  testingSamples, world_size, world_rank);
        double startTime = MPI_Wtime();
        if (optArgs.isIsNormalTime()) {
            sgd1.adamSGD(w);
        }

        if (optArgs.isIsEpochTime()) {
            string suffix = getTimeStamp();
            logfile.append(logsourceBase).append("logs/epochlog/").append(datasource).append("/").append(
                    "world_size=").append(to_string(world_size)).append("_iterations=").append(
                    to_string(optArgs.getIterations()));
            logfile.append("_").append(suffix);
            sgd1.adamSGD(w, logfile);
        }


        double endTime = MPI_Wtime();
        if (world_rank == 0) {
            cout << "Training Time : " << (endTime - startTime) << endl;
            Predict predict(Xtest, ytest, w, testSet, features);
            double acc = predict.predict();
            cout << "Testing Accuracy : " << acc << "%" << endl;
            summary(summarylogfile, world_size, acc, (endTime - startTime), datasource);

        }

        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;
    } else {
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string logsourceBase = resourceManager.getLogSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string trainFile;
        trainFile.append(datasourceBase).append(datasource).append(trainFileName);
        string testFile;
        testFile.append(datasourceBase).append(datasource).append(testFileName);
        cout << "Train File : " << trainFile << endl;
        cout << "Test File : " << testFile << endl;
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples;
        int testSet = testingSamples;
        int dataPerMachine = trainSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;

        double *w = new double[features];

        double ytrain[dataPerMachine];
        initializer.initializeWeightsWithArray(dataPerMachine, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[dataPerMachine];
        for (int i = 0; i < dataPerMachine; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        DataSet dataSet(features, trainingSamples, testingSamples, trainFile, testFile, world_size, world_rank, optArgs.isBulk());
        dataSet.distributedLoad(Xtrain, ytrain, Xtest, ytest);
//        dataPerMachine = dataSet.getDataPerMachine();
//        if(world_rank==0) {
//            cout << "From Main : " << "Data Per Machine : " << dataPerMachine << endl;
//            Util util;
//            util.print2DMatrix(Xtest, 20, features);
//            printf("\n----------------------------------------\n");
//        }


        PSGD sgd1(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, dataPerMachine,
                  testingSamples, world_size, world_rank);
        double startTime = MPI_Wtime();
        if (optArgs.isIsNormalTime()) {
            sgd1.adamSGD(w);
        }

        if (optArgs.isIsEpochTime()) {
            string suffix = getTimeStamp();
            logfile.append(logsourceBase).append("logs/epochlog/").append(datasource).append("/").append(
                    "world_size=").append(to_string(world_size)).append("_iterations=").append(
                    to_string(optArgs.getIterations()));
            logfile.append("_").append(suffix);
            sgd1.adamSGD(w, logfile);
        }


        double endTime = MPI_Wtime();
        if (world_rank == 0) {

            double acc = 0;
            if(!optArgs.isBulk()){
                Predict predict(Xtest, ytest, w, testSet, features);
                acc = predict.predict();
            }else{
                string bulktestfile;
                bulktestfile.append(datasourceBase).append(datasource).append("/bulk/");
                string dir = string(bulktestfile);
                vector<string> files = vector<string>();

                getdir(dir,files);
                int fixedTest = 20000;
                double ytest[fixedTest];
                initializer.initializeWeightsWithArray(testSet, ytest);

                double **Xtest;

                Xtest = new double *[fixedTest];
                for (int i = 0; i < fixedTest; ++i) {
                    Xtest[i] = new double[features];
                }
                double cum_acc = 0;
                for (unsigned int i = 0;i < files.size()-1;i++) {
                    string file="";
                    file.append(bulktestfile).append(files[i]);
                    cout << file << endl;
                    DataSet dataSet1(features, fixedTest,file);
                    dataSet1.loadTestData(Xtest, ytest);
                    Predict predict(Xtest, ytest, w, testSet, features);
                    acc = predict.predict();
                    cout << "Test " << i << ", Accuracy : " << acc << endl;
                    cum_acc += acc;
                }
                acc = cum_acc / double(files.size());
            }
            cout << "Training Time : " << (endTime - startTime) << endl;
            cout << "Testing Accuracy : " << acc << "%" << endl;
            summary(summarylogfile, world_size, acc, (endTime - startTime), datasource);
        }

        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;
    }
    MPI_Finalize();
}

void sequentialLoad(OptArgs optArgs) {
    int world_rank=0;
    int world_size=1;
    optArgs.toString();
    ResourceManager resourceManager;
    Initializer initializer;
    resourceManager.loadDataSourcePath();
    resourceManager.loadLogSourcePath();
    resourceManager.loadSummaryPath();
    string summarylogfile ="";
    summarylogfile.append(resourceManager.getLogSummaryBasePath()).append("/").append(optArgs.getDataset()).append("/summary.csv");
    string logfile = "";
    if (optArgs.isIsSplit()) {
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string logsourceBase = resourceManager.getLogSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string sourceFile;
        sourceFile.append(datasourceBase).append(datasource).append(trainFileName);
        cout << "SourceFile : " << sourceFile << endl;
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples * ratio;
        int testSet = totalSamples - trainSet;
        int dataPerMachine = trainSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;

        double *w = new double[features];

        double ytrain[dataPerMachine];
        initializer.initializeWeightsWithArray(dataPerMachine, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[dataPerMachine];
        for (int i = 0; i < dataPerMachine; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        DataSet dataSet(features, trainingSamples, testingSamples, optArgs.isIsSplit(), optArgs.getRatio(), sourceFile,
                        world_size, world_rank);
        dataSet.distributedLoad(Xtrain, ytrain, Xtest, ytest);
//        dataPerMachine = dataSet.getDataPerMachine();
//        if(world_rank==0) {
//            cout << "From Main : " << "Data Per Machine : " << dataPerMachine << endl;
//            Util util;
//            util.print2DMatrix(Xtest, 20, features);
//            printf("\n----------------------------------------\n");
//        }


        PSGD sgd1(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, dataPerMachine,
                  testingSamples, world_size, world_rank);
        double startTime = MPI_Wtime();
        if (optArgs.isIsNormalTime()) {
            sgd1.adamSGDSeq(w);
        }

        if (optArgs.isIsEpochTime()) {
            string suffix = getTimeStamp();
            logfile.append(logsourceBase).append("logs/epochlog/").append(datasource).append("/").append(
                    "world_size=").append(to_string(world_size)).append("_iterations=").append(
                    to_string(optArgs.getIterations()));
            logfile.append("_").append(suffix);
            sgd1.adamSGDSeq(w, logfile);
        }


        double endTime = MPI_Wtime();
        if (world_rank == 0) {
            cout << "Training Time : " << (endTime - startTime) << endl;
            Predict predict(Xtest, ytest, w, testSet, features);
            double acc = predict.predict();
            cout << "Testing Accuracy : " << acc << "%" << endl;
            summary(summarylogfile, world_size, acc, (endTime - startTime), datasource);

        }

        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;
    } else {
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string logsourceBase = resourceManager.getLogSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string trainFile;
        trainFile.append(datasourceBase).append(datasource).append(trainFileName);
        string testFile;
        testFile.append(datasourceBase).append(datasource).append(testFileName);
        cout << "Train File : " << trainFile << endl;
        cout << "Test File : " << testFile << endl;
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples;
        int testSet = testingSamples;
        int dataPerMachine = trainSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;

        double *w = new double[features];

        double ytrain[dataPerMachine];
        initializer.initializeWeightsWithArray(dataPerMachine, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[dataPerMachine];
        for (int i = 0; i < dataPerMachine; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        DataSet dataSet(features, trainingSamples, testingSamples, trainFile, testFile, world_size, world_rank, optArgs.isBulk());
        dataSet.distributedLoad(Xtrain, ytrain, Xtest, ytest);
//        dataPerMachine = dataSet.getDataPerMachine();
//        if(world_rank==0) {
//            cout << "From Main : " << "Data Per Machine : " << dataPerMachine << endl;
//            Util util;
//            util.print2DMatrix(Xtest, 20, features);
//            printf("\n----------------------------------------\n");
//        }


        PSGD sgd1(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, dataPerMachine,
                  testingSamples, world_size, world_rank);
        double startTime = MPI_Wtime();
        if (optArgs.isIsNormalTime()) {
            sgd1.adamSGDSeq(w);
        }

        if (optArgs.isIsEpochTime()) {
            string suffix = getTimeStamp();
            logfile.append(logsourceBase).append("logs/epochlog/").append(datasource).append("/").append(
                    "sequential_world_size=").append(to_string(world_size)).append("_iterations=").append(
                    to_string(optArgs.getIterations()));
            logfile.append("_").append(suffix);
            sgd1.adamSGDSeq(w);
        }


        double endTime = MPI_Wtime();
        if (world_rank == 0) {

            double acc = 0;
            if(!optArgs.isBulk()){
                Predict predict(Xtest, ytest, w, testSet, features);
                acc = predict.predict();
            }else{
                string bulktestfile;
                bulktestfile.append(datasourceBase).append(datasource).append("/bulk/");
                string dir = string(bulktestfile);
                vector<string> files = vector<string>();

                getdir(dir,files);
                int fixedTest = 20000;
                double ytest[fixedTest];
                initializer.initializeWeightsWithArray(testSet, ytest);

                double **Xtest;

                Xtest = new double *[fixedTest];
                for (int i = 0; i < fixedTest; ++i) {
                    Xtest[i] = new double[features];
                }
                double cum_acc = 0;
                for (unsigned int i = 0;i < files.size()-1;i++) {
                    string file="";
                    file.append(bulktestfile).append(files[i]);
                    cout << file << endl;
                    DataSet dataSet1(features, fixedTest,file);
                    dataSet1.loadTestData(Xtest, ytest);
                    Predict predict(Xtest, ytest, w, testSet, features);
                    acc = predict.predict();
                    cout << "Test " << i << ", Accuracy : " << acc << endl;
                    cum_acc += acc;
                }
                acc = cum_acc / double(files.size());
            }
            cout << "Training Time : " << (endTime - startTime) << endl;
            cout << "Testing Accuracy : " << acc << "%" << endl;
            summary(summarylogfile, world_size, acc, (endTime - startTime), datasource);
        }

        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;
    }

}

void parallelLoadBatchV1(OptArgs optArgs) {
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    optArgs.toString();
    ResourceManager resourceManager;
    Initializer initializer;
    resourceManager.loadDataSourcePath();
    resourceManager.loadLogSourcePath();
    resourceManager.loadSummaryPath();
    string summarylogfile ="";
    summarylogfile.append(resourceManager.getLogSummaryBasePath()).append("/").append(optArgs.getDataset()).append("/summary.csv");
    string logfile = "";
    if (optArgs.isIsSplit()) {
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string logsourceBase = resourceManager.getLogSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string sourceFile;
        sourceFile.append(datasourceBase).append(datasource).append(trainFileName);
        cout << "SourceFile : " << sourceFile << endl;
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples * ratio;
        int testSet = totalSamples - trainSet;
        int dataPerMachine = trainSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;

        double *w = new double[features];

        double ytrain[dataPerMachine];
        initializer.initializeWeightsWithArray(dataPerMachine, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[dataPerMachine];
        for (int i = 0; i < dataPerMachine; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        DataSet dataSet(features, trainingSamples, testingSamples, optArgs.isIsSplit(), optArgs.getRatio(), sourceFile,
                        world_size, world_rank);
        dataSet.distributedLoad(Xtrain, ytrain, Xtest, ytest);
//        dataPerMachine = dataSet.getDataPerMachine();
//        if(world_rank==0) {
//            cout << "From Main : " << "Data Per Machine : " << dataPerMachine << endl;
//            Util util;
//            util.print2DMatrix(Xtest, 20, features);
//            printf("\n----------------------------------------\n");
//        }


        PSGD sgd1(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, dataPerMachine,
                  testingSamples, world_size, world_rank);
        double startTime = MPI_Wtime();
        if (optArgs.isIsNormalTime()) {
            sgd1.adamSGDBatchv1(w);
        }

        if (optArgs.isIsEpochTime()) {
            string suffix = getTimeStamp();
            logfile.append(logsourceBase).append("logs/epochlog/").append(datasource).append("/").append(
                    "world_size=").append(to_string(world_size)).append("_iterations=").append(
                    to_string(optArgs.getIterations()));
            logfile.append("_").append(suffix);
            sgd1.adamSGDBatchv1(w, logfile);
        }


        double endTime = MPI_Wtime();
        if (world_rank == 0) {
            cout << "Training Time : " << (endTime - startTime) << endl;
            Predict predict(Xtest, ytest, w, testSet, features);
            double acc = predict.predict();
            cout << "Testing Accuracy : " << acc << "%" << endl;
            summary(summarylogfile, world_size, acc, (endTime - startTime), datasource);

        }

        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;
    } else {
        if(world_rank==0) {
            cout << "Training with full data set..." << endl;
        }
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string logsourceBase = resourceManager.getLogSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string trainFile;
        trainFile.append(datasourceBase).append(datasource).append(trainFileName);
        string testFile;
        testFile.append(datasourceBase).append(datasource).append(testFileName);
        cout << "Train File : " << trainFile << endl;
        cout << "Test File : " << testFile << endl;
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples;
        int testSet = testingSamples;
        int dataPerMachine = trainSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;

        double *w = new double[features];

        double ytrain[dataPerMachine];
        initializer.initializeWeightsWithArray(dataPerMachine, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[dataPerMachine];
        for (int i = 0; i < dataPerMachine; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        DataSet dataSet(features, trainingSamples, testingSamples, trainFile, testFile, world_size, world_rank, optArgs.isBulk());
        dataSet.distributedLoad(Xtrain, ytrain, Xtest, ytest);
//        dataPerMachine = dataSet.getDataPerMachine();
//        if(world_rank==0) {
//            cout << "From Main : " << "Data Per Machine : " << dataPerMachine << endl;
//            Util util;
//            util.print2DMatrix(Xtest, 20, features);
//            printf("\n----------------------------------------\n");
//        }


        PSGD sgd1(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, dataPerMachine,
                  testingSamples, world_size, world_rank);
        double startTime = MPI_Wtime();
        if (optArgs.isIsNormalTime()) {
            sgd1.adamSGDBatchv1(w);
        }

        if (optArgs.isIsEpochTime()) {
            string suffix = getTimeStamp();
            logfile.append(logsourceBase).append("logs/epochlog/").append(datasource).append("/").append(
                    "world_size=").append(to_string(world_size)).append("_iterations=").append(
                    to_string(optArgs.getIterations()));
            logfile.append("_").append(suffix);
            sgd1.adamSGD(w, logfile);
        }


        double endTime = MPI_Wtime();
        if (world_rank == 0) {

            double acc = 0;
            if(!optArgs.isBulk()){
                Predict predict(Xtest, ytest, w, testSet, features);
                acc = predict.predict();
            }else{
                string bulktestfile;
                bulktestfile.append(datasourceBase).append(datasource).append("/bulk/");
                string dir = string(bulktestfile);
                vector<string> files = vector<string>();

                getdir(dir,files);
                int fixedTest = 20000;
                double ytest[fixedTest];
                initializer.initializeWeightsWithArray(testSet, ytest);

                double **Xtest;

                Xtest = new double *[fixedTest];
                for (int i = 0; i < fixedTest; ++i) {
                    Xtest[i] = new double[features];
                }
                double cum_acc = 0;
                for (unsigned int i = 0;i < files.size()-1;i++) {
                    string file="";
                    file.append(bulktestfile).append(files[i]);
                    cout << file << endl;
                    DataSet dataSet1(features, fixedTest,file);
                    dataSet1.loadTestData(Xtest, ytest);
                    Predict predict(Xtest, ytest, w, testSet, features);
                    acc = predict.predict();
                    cout << "Test " << i << ", Accuracy : " << acc << endl;
                    cum_acc += acc;
                }
                acc = cum_acc / double(files.size());
            }
            cout << "Training Time : " << (endTime - startTime) << endl;
            cout << "Testing Accuracy : " << acc << "%" << endl;
            summary(summarylogfile, world_size, acc, (endTime - startTime), datasource);
        }

        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;
    }
    MPI_Finalize();
}

void parallelLoadBatchV2(OptArgs optArgs, int comm_gap) {
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //optArgs.toString();
    ResourceManager resourceManager;
    Initializer initializer;
    Util util;
    resourceManager.loadDataSourcePath();
    resourceManager.loadLogSourcePath();
    resourceManager.loadSummaryPath();
    resourceManager.loadWeightSummaryPath();
    resourceManager.loadEpochSummaryPath();
    string exptype = "adam";
    string summarylogfile ="";
    summarylogfile.append(resourceManager.getLogSummaryBasePath()).append("/parallel/").append(exptype).append("/batch/").append(optArgs.getDataset()).append("/").append("summary_comm_gap=").append(to_string(optArgs.getBatch_per())).append(".csv");
    string weightlogfile = "";
    weightlogfile.append(resourceManager.getWeightSummaryBasePath()).append("/").append(optArgs.getDataset()).append("/").append(getTimeStamp())
            .append("_").append("batch_weight_summary.csv");
    string epochlogfile = resourceManager.getEpochlogSummaryBasePath();
    epochlogfile.append("parallel/").append(exptype).append("/batch/").append(optArgs.getDataset()).append("/").append(getTimeStamp()).append("_rank_").append(to_string(world_rank)).append("_batch_cross_validation_accuracy.csv");
    string logfile = "";

    if (optArgs.isIsSplit()) {
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string logsourceBase = resourceManager.getLogSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string sourceFile;
        sourceFile.append(datasourceBase).append(datasource).append(trainFileName);
        cout << "SourceFile : " << sourceFile << endl;
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples * ratio;
        int testSet = totalSamples - trainSet;
        int dataPerMachine = trainSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;
        comm_gap = optArgs.getBatch_per() * dataPerMachine;
        if(world_rank==0) {
            cout << "Comm Gap : " << comm_gap << endl;
        }
        double *w = new double[features];

        double ytrain[dataPerMachine];
        initializer.initializeWeightsWithArray(dataPerMachine, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[dataPerMachine];
        for (int i = 0; i < dataPerMachine; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        DataSet dataSet(features, trainingSamples, testingSamples, optArgs.isIsSplit(), optArgs.getRatio(), sourceFile,
                        world_size, world_rank);
        dataSet.distributedLoad(Xtrain, ytrain, Xtest, ytest);
//        dataPerMachine = dataSet.getDataPerMachine();
//        if(world_rank==0) {
//            cout << "From Main : " << "Data Per Machine : " << dataPerMachine << endl;
//            Util util;
//            util.print2DMatrix(Xtest, 20, features);
//            printf("\n----------------------------------------\n");
//        }


        PSGD sgd1(0.5, 0.992, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, dataPerMachine,
                  testingSamples, world_size, world_rank, Xtest, ytest);
        double startTime = MPI_Wtime();
        if (optArgs.isIsNormalTime()) {
            sgd1.adamSGDBatchv2(w, comm_gap, logfile, epochlogfile);
        }

        if (optArgs.isIsEpochTime()) {
            string suffix = getTimeStamp();
            logfile.append(logsourceBase).append("logs/epochlog/").append(datasource).append("/").append(
                    "world_size=").append(to_string(world_size)).append("_iterations=").append(
                    to_string(optArgs.getIterations()));
            logfile.append("_").append(suffix);
            sgd1.adamSGDBatchv2(w, comm_gap, logfile);
        }


        double endTime = MPI_Wtime();
        if (world_rank == 0) {
            cout << "Training Time : " << (endTime - startTime) << endl;
            Predict predict(Xtest, ytest, w, testSet, features);
            double acc = predict.predict();
            cout << "Testing Accuracy : " << acc << "%" << endl;
            summary(summarylogfile, world_size, acc, (endTime - startTime), datasource);
            util.writeWeight(w, features, weightlogfile);
        }

        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;
    } else {
        if(world_rank==0) {
            cout << "Training with full data set..." << endl;
        }
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string logsourceBase = resourceManager.getLogSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string trainFile;
        trainFile.append(datasourceBase).append(datasource).append(trainFileName);
        string testFile;
        testFile.append(datasourceBase).append(datasource).append(testFileName);
        cout << "Train File : " << trainFile << endl;
        cout << "Test File : " << testFile << endl;
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples;
        int testSet = testingSamples;
        int dataPerMachine = trainSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;
        comm_gap = optArgs.getBatch_per() * dataPerMachine;
        if(world_rank==0) {
            cout << "Comm Gap : " << comm_gap << endl;
        }
        double *w = new double[features];

        double ytrain[dataPerMachine];
        initializer.initializeWeightsWithArray(dataPerMachine, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[dataPerMachine];
        for (int i = 0; i < dataPerMachine; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        DataSet dataSet(features, trainingSamples, testingSamples, trainFile, testFile, world_size, world_rank, optArgs.isBulk());
        dataSet.distributedLoad(Xtrain, ytrain, Xtest, ytest);
//        dataPerMachine = dataSet.getDataPerMachine();
//        if(world_rank==0) {
//            cout << "From Main : " << "Data Per Machine : " << dataPerMachine << endl;
//            Util util;
//            util.print2DMatrix(Xtest, 20, features);
//            printf("\n----------------------------------------\n");
//        }


        PSGD sgd1(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, dataPerMachine,
                  testingSamples, world_size, world_rank);
        double startTime = MPI_Wtime();
        if (optArgs.isIsNormalTime()) {
            sgd1.adamSGDBatchv2(w, comm_gap, logfile, epochlogfile);
            //sgd1.adamSGDBatchv2(w, comm_gap);
        }

        if (optArgs.isIsEpochTime()) {
            string suffix = getTimeStamp();
            logfile.append(logsourceBase).append("logs/epochlog/").append(datasource).append("/").append(
                    "world_size=").append(to_string(world_size)).append("_iterations=").append(
                    to_string(optArgs.getIterations()));
            logfile.append("_").append(suffix);
            sgd1.adamSGDBatchv2(w, comm_gap, logfile);
        }


        double endTime = MPI_Wtime();
        if (world_rank == 0) {

            double acc = 0;
            if(!optArgs.isBulk()){
                Predict predict(Xtest, ytest, w, testSet, features);
                acc = predict.predict();
            }else{
                string bulktestfile;
                bulktestfile.append(datasourceBase).append(datasource).append("/bulk/");
                string dir = string(bulktestfile);
                vector<string> files = vector<string>();

                getdir(dir,files);
                int fixedTest = 20000;
                double ytest[fixedTest];
                initializer.initializeWeightsWithArray(testSet, ytest);

                double **Xtest;

                Xtest = new double *[fixedTest];
                for (int i = 0; i < fixedTest; ++i) {
                    Xtest[i] = new double[features];
                }
                double cum_acc = 0;
                for (unsigned int i = 0;i < files.size()-1;i++) {
                    string file="";
                    file.append(bulktestfile).append(files[i]);
                    cout << file << endl;
                    DataSet dataSet1(features, fixedTest,file);
                    dataSet1.loadTestData(Xtest, ytest);
                    Predict predict(Xtest, ytest, w, testSet, features);
                    acc = predict.predict();
                    cout << "Test " << i << ", Accuracy : " << acc << endl;
                    cum_acc += acc;
                }
                acc = cum_acc / double(files.size());
            }
            cout << "Training Time : " << (endTime - startTime) << endl;
            cout << "Testing Accuracy : " << acc << "%" << endl;
            summary(summarylogfile, world_size, acc, (endTime - startTime), datasource);
            util.writeWeight(w, features, weightlogfile);
        }

        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;
    }
    MPI_Finalize();
}

void parallelLoadRotationV1(OptArgs optArgs) {
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //optArgs.toString();
    ResourceManager resourceManager;
    Initializer initializer;
    Util util;
    resourceManager.loadDataSourcePath();
    resourceManager.loadLogSourcePath();
    resourceManager.loadSummaryPath();
    resourceManager.loadWeightSummaryPath();
    string summarylogfile ="";
    summarylogfile.append(resourceManager.getLogSummaryBasePath()).append("/").append(optArgs.getDataset()).append("/summary.csv");
    string weightlogfile = "";
    weightlogfile.append(resourceManager.getWeightSummaryBasePath()).append("/").append(optArgs.getDataset()).append("/").append(getTimeStamp())
            .append("_").append("ring_weight_summary.csv");
    if(world_rank==0) {
        cout << weightlogfile << endl;
    }
    string logfile = "";
    if (optArgs.isIsSplit()) {
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string logsourceBase = resourceManager.getLogSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string sourceFile;
        sourceFile.append(datasourceBase).append(datasource).append(trainFileName);
        if(world_rank==0) {
            cout << "SourceFile : " << sourceFile << endl;
        }

        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples * ratio;
        int testSet = totalSamples - trainSet;
        int dataPerMachine = trainSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;

        double *w = new double[features];

        double ytrain[dataPerMachine];
        initializer.initializeWeightsWithArray(dataPerMachine, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[dataPerMachine];
        for (int i = 0; i < dataPerMachine; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        DataSet dataSet(features, trainingSamples, testingSamples, optArgs.isIsSplit(), optArgs.getRatio(), sourceFile,
                        world_size, world_rank);
        dataSet.distributedLoad(Xtrain, ytrain, Xtest, ytest);
//        dataPerMachine = dataSet.getDataPerMachine();
//        if(world_rank==0) {
//            cout << "From Main : " << "Data Per Machine : " << dataPerMachine << endl;
//            Util util;
//            util.print2DMatrix(Xtest, 20, features);
//            printf("\n----------------------------------------\n");
//        }


        PSGD sgd1(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, dataPerMachine,
                  testingSamples, world_size, world_rank);
        double startTime = MPI_Wtime();
        if (optArgs.isIsNormalTime()) {
            sgd1.adamSGDRotationv1(w);
        }

        if (optArgs.isIsEpochTime()) {
            string suffix = getTimeStamp();
            logfile.append(logsourceBase).append("logs/epochlog/").append(datasource).append("/").append(
                    "world_size=").append(to_string(world_size)).append("_iterations=").append(
                    to_string(optArgs.getIterations()));
            logfile.append("_").append(suffix);
            sgd1.adamSGDBatchv1(w, logfile);
        }


        double endTime = MPI_Wtime();
        if (world_rank == 0) {
            cout << "Training Time : " << (endTime - startTime) << endl;
            Predict predict(Xtest, ytest, w, testSet, features);
            double acc = predict.predict();
            cout << "Testing Accuracy : " << acc << "%" << endl;
            summary(summarylogfile, world_size, acc, (endTime - startTime), datasource);
            util.writeWeight(w, features, weightlogfile);
        }

        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;
    } else {
        if(world_rank==0) {
            cout << "Training with full data set..." << endl;
        }
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string logsourceBase = resourceManager.getLogSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string trainFile;
        trainFile.append(datasourceBase).append(datasource).append(trainFileName);
        string testFile;
        testFile.append(datasourceBase).append(datasource).append(testFileName);
        cout << "Train File : " << trainFile << endl;
        cout << "Test File : " << testFile << endl;
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples;
        int testSet = testingSamples;
        int dataPerMachine = trainSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;

        double *w = new double[features];

        double ytrain[dataPerMachine];
        initializer.initializeWeightsWithArray(dataPerMachine, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[dataPerMachine];
        for (int i = 0; i < dataPerMachine; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        DataSet dataSet(features, trainingSamples, testingSamples, trainFile, testFile, world_size, world_rank, optArgs.isBulk());
        dataSet.distributedLoad(Xtrain, ytrain, Xtest, ytest);
//        dataPerMachine = dataSet.getDataPerMachine();
//        if(world_rank==0) {
//            cout << "From Main : " << "Data Per Machine : " << dataPerMachine << endl;
//            Util util;
//            util.print2DMatrix(Xtest, 20, features);
//            printf("\n----------------------------------------\n");
//        }


        PSGD sgd1(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, dataPerMachine,
                  testingSamples, world_size, world_rank);
        double startTime = MPI_Wtime();
        if (optArgs.isIsNormalTime()) {
            sgd1.adamSGDRotationv1(w);
        }

        if (optArgs.isIsEpochTime()) {
            string suffix = getTimeStamp();
            logfile.append(logsourceBase).append("logs/epochlog/").append(datasource).append("/").append(
                    "world_size=").append(to_string(world_size)).append("_iterations=").append(
                    to_string(optArgs.getIterations()));
            logfile.append("_").append(suffix);
            sgd1.adamSGD(w, logfile);
        }


        double endTime = MPI_Wtime();
        if (world_rank == 0) {

            double acc = 0;
            if(!optArgs.isBulk()){
                Predict predict(Xtest, ytest, w, testSet, features);
                acc = predict.predict();
            }else{
                string bulktestfile;
                bulktestfile.append(datasourceBase).append(datasource).append("/bulk/");
                string dir = string(bulktestfile);
                vector<string> files = vector<string>();

                getdir(dir,files);
                int fixedTest = 20000;
                double ytest[fixedTest];
                initializer.initializeWeightsWithArray(testSet, ytest);

                double **Xtest;

                Xtest = new double *[fixedTest];
                for (int i = 0; i < fixedTest; ++i) {
                    Xtest[i] = new double[features];
                }
                double cum_acc = 0;
                for (unsigned int i = 0;i < files.size()-1;i++) {
                    string file="";
                    file.append(bulktestfile).append(files[i]);
                    cout << file << endl;
                    DataSet dataSet1(features, fixedTest,file);
                    dataSet1.loadTestData(Xtest, ytest);
                    Predict predict(Xtest, ytest, w, testSet, features);
                    acc = predict.predict();
                    cout << "Test " << i << ", Accuracy : " << acc << endl;
                    cum_acc += acc;
                }
                acc = cum_acc / double(files.size());
            }
            cout << "Training Time : " << (endTime - startTime) << endl;
            cout << "Testing Accuracy : " << acc << "%" << endl;
            summary(summarylogfile, world_size, acc, (endTime - startTime), datasource);
            util.writeWeight(w, features, weightlogfile);
        }

        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;
    }
    MPI_Finalize();
}


void parallelLoadRandomV1(OptArgs optArgs) {

    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //optArgs.toString();
    ResourceManager resourceManager;
    Initializer initializer;
    resourceManager.loadDataSourcePath();
    resourceManager.loadLogSourcePath();
    resourceManager.loadSummaryPath();
    string summarylogfile ="";
    summarylogfile.append(resourceManager.getLogSummaryBasePath()).append("/").append(optArgs.getDataset()).append("/summary.csv");
    string weightlogfile="";

    string logfile = "";
    if(world_rank==0) {
        cout << "Random Ring version 1 " << endl;
    }
    if (optArgs.isIsSplit()) {
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string logsourceBase = resourceManager.getLogSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string sourceFile;
        sourceFile.append(datasourceBase).append(datasource).append(trainFileName);
        if(world_rank==0) {
            cout << "SourceFile : " << sourceFile << endl;
        }

        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples * ratio;
        int testSet = totalSamples - trainSet;
        int dataPerMachine = trainSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;

        double *w = new double[features];

        double ytrain[dataPerMachine];
        initializer.initializeWeightsWithArray(dataPerMachine, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[dataPerMachine];
        for (int i = 0; i < dataPerMachine; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        DataSet dataSet(features, trainingSamples, testingSamples, optArgs.isIsSplit(), optArgs.getRatio(), sourceFile,
                        world_size, world_rank);
        dataSet.distributedLoad(Xtrain, ytrain, Xtest, ytest);
//        dataPerMachine = dataSet.getDataPerMachine();
//        if(world_rank==0) {
//            cout << "From Main : " << "Data Per Machine : " << dataPerMachine << endl;
//            Util util;
//            util.print2DMatrix(Xtest, 20, features);
//            printf("\n----------------------------------------\n");
//        }


        PSGD sgd1(0.55, 0.55, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, dataPerMachine,
                  testingSamples, world_size, world_rank);
        double startTime = MPI_Wtime();
        if (optArgs.isIsNormalTime()) {
            if(optArgs.isIsDrop()) {
                double drop_per = optArgs.getDrop_out_per();
                sgd1.adamSGDRandomRingv1(w, drop_per, summarylogfile);
            }

        }

        if (optArgs.isIsEpochTime()) {
            string suffix = getTimeStamp();
            logfile.append(logsourceBase).append("logs/epochlog/").append(datasource).append("/").append(
                    "world_size=").append(to_string(world_size)).append("_iterations=").append(
                    to_string(optArgs.getIterations()));
            logfile.append("_").append(suffix);
            sgd1.adamSGDBatchv1(w, logfile);
        }


        double endTime = MPI_Wtime();
        if (world_rank == 0) {
            cout << "Training Time : " << (endTime - startTime) << endl;
            Predict predict(Xtest, ytest, w, testSet, features);
            double acc = predict.predict();
            cout << "Testing Accuracy : " << acc << "%" << endl;
            summary(summarylogfile, world_size, acc, (endTime - startTime), datasource);
        }

        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;
    } else {
        if(world_rank==0) {
            cout << "Training with full data set..." << endl;
        }
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string logsourceBase = resourceManager.getLogSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string trainFile;
        trainFile.append(datasourceBase).append(datasource).append(trainFileName);
        string testFile;
        testFile.append(datasourceBase).append(datasource).append(testFileName);
        cout << "Train File : " << trainFile << endl;
        cout << "Test File : " << testFile << endl;
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples;
        int testSet = testingSamples;
        int dataPerMachine = trainSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;

        double *w = new double[features];

        double ytrain[dataPerMachine];
        initializer.initializeWeightsWithArray(dataPerMachine, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[dataPerMachine];
        for (int i = 0; i < dataPerMachine; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        DataSet dataSet(features, trainingSamples, testingSamples, trainFile, testFile, world_size, world_rank, optArgs.isBulk());
        dataSet.distributedLoad(Xtrain, ytrain, Xtest, ytest);
//        dataPerMachine = dataSet.getDataPerMachine();
//        if(world_rank==0) {
//            cout << "From Main : " << "Data Per Machine : " << dataPerMachine << endl;
//            Util util;
//            util.print2DMatrix(Xtest, 20, features);
//            printf("\n----------------------------------------\n");
//        }


        PSGD sgd1(0.55, 0.55, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, dataPerMachine,
                  testingSamples, world_size, world_rank);
        double startTime = MPI_Wtime();
        if (optArgs.isIsNormalTime()) {
            double drop_per = optArgs.getDrop_out_per();
            sgd1.adamSGDRandomRingv1(w, drop_per, summarylogfile);
        }

        if (optArgs.isIsEpochTime()) {
            string suffix = getTimeStamp();
            logfile.append(logsourceBase).append("logs/epochlog/").append(datasource).append("/").append(
                    "world_size=").append(to_string(world_size)).append("_iterations=").append(
                    to_string(optArgs.getIterations()));
            logfile.append("_").append(suffix);
            sgd1.adamSGD(w, logfile);
        }


        double endTime = MPI_Wtime();
        if (world_rank == 0) {

            double acc = 0;
            if(!optArgs.isBulk()){
                Predict predict(Xtest, ytest, w, testSet, features);
                acc = predict.predict();
            }else{
                string bulktestfile;
                bulktestfile.append(datasourceBase).append(datasource).append("/bulk/");
                string dir = string(bulktestfile);
                vector<string> files = vector<string>();

                getdir(dir,files);
                int fixedTest = 20000;
                double ytest[fixedTest];
                initializer.initializeWeightsWithArray(testSet, ytest);

                double **Xtest;

                Xtest = new double *[fixedTest];
                for (int i = 0; i < fixedTest; ++i) {
                    Xtest[i] = new double[features];
                }
                double cum_acc = 0;
                for (unsigned int i = 0;i < files.size()-1;i++) {
                    string file="";
                    file.append(bulktestfile).append(files[i]);
                    cout << file << endl;
                    DataSet dataSet1(features, fixedTest,file);
                    dataSet1.loadTestData(Xtest, ytest);
                    Predict predict(Xtest, ytest, w, testSet, features);
                    acc = predict.predict();
                    cout << "Test " << i << ", Accuracy : " << acc << endl;
                    cum_acc += acc;
                }
                acc = cum_acc / double(files.size());
            }
            cout << "Training Time : " << (endTime - startTime) << endl;
            cout << "Testing Accuracy : " << acc << "%" << endl;
            summary(summarylogfile, world_size, acc, (endTime - startTime), datasource);
        }

        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;
    }
    MPI_Finalize();
}

void parallelLoadRandomV2(OptArgs optArgs) {
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //optArgs.toString();
    if(world_rank==0) {
        cout << "Random Ring v2" << endl;
    }
    ResourceManager resourceManager;
    Initializer initializer;
    resourceManager.loadDataSourcePath();
    resourceManager.loadLogSourcePath();
    resourceManager.loadSummaryPath();
    string summarylogfile ="";
    summarylogfile.append(resourceManager.getLogSummaryBasePath()).append("/").append(optArgs.getDataset()).append("/summary.csv");
    string logfile = "";

    if (optArgs.isIsSplit()) {
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string logsourceBase = resourceManager.getLogSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string sourceFile;
        sourceFile.append(datasourceBase).append(datasource).append(trainFileName);
        if(world_rank==0) {
            cout << "SourceFile : " << sourceFile << endl;
        }
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples * ratio;
        int testSet = totalSamples - trainSet;
        int dataPerMachine = trainSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;

        double *w = new double[features];

        double ytrain[dataPerMachine];
        initializer.initializeWeightsWithArray(dataPerMachine, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[dataPerMachine];
        for (int i = 0; i < dataPerMachine; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        DataSet dataSet(features, trainingSamples, testingSamples, optArgs.isIsSplit(), optArgs.getRatio(), sourceFile,
                        world_size, world_rank);
        dataSet.distributedLoad(Xtrain, ytrain, Xtest, ytest);
//        dataPerMachine = dataSet.getDataPerMachine();
//        if(world_rank==0) {
//            cout << "From Main : " << "Data Per Machine : " << dataPerMachine << endl;
//            Util util;
//            util.print2DMatrix(Xtest, 20, features);
//            printf("\n----------------------------------------\n");
//        }


        PSGD sgd1(0.50, 0.50, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, dataPerMachine,
                  testingSamples, world_size, world_rank);
        double startTime = MPI_Wtime();
        if (optArgs.isIsNormalTime()) {
            if(optArgs.isIsDrop()) {
                double drop_per = optArgs.getDrop_out_per();
                sgd1.adamSGDRandomRingv2(w, drop_per, summarylogfile);
            }

        }

        if (optArgs.isIsEpochTime()) {
            string suffix = getTimeStamp();
            logfile.append(logsourceBase).append("logs/epochlog/").append(datasource).append("/").append(
                    "world_size=").append(to_string(world_size)).append("_iterations=").append(
                    to_string(optArgs.getIterations()));
            logfile.append("_").append(suffix);
            sgd1.adamSGDBatchv1(w, logfile);
        }


        double endTime = MPI_Wtime();
        if (world_rank == 0) {
            cout << "Training Time : " << (endTime - startTime) << endl;
            Predict predict(Xtest, ytest, w, testSet, features);
            double acc = predict.predict();
            cout << "Testing Accuracy : " << acc << "%" << endl;
            summary(summarylogfile, world_size, acc, (endTime - startTime), datasource);
        }

        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;
    } else {
        if(world_rank==0) {
            cout << "Training with full data set..." << endl;
        }
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string logsourceBase = resourceManager.getLogSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string trainFile;
        trainFile.append(datasourceBase).append(datasource).append(trainFileName);
        string testFile;
        testFile.append(datasourceBase).append(datasource).append(testFileName);
        cout << "Train File : " << trainFile << endl;
        cout << "Test File : " << testFile << endl;
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples;
        int testSet = testingSamples;
        int dataPerMachine = trainSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;

        double *w = new double[features];

        double ytrain[dataPerMachine];
        initializer.initializeWeightsWithArray(dataPerMachine, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[dataPerMachine];
        for (int i = 0; i < dataPerMachine; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        DataSet dataSet(features, trainingSamples, testingSamples, trainFile, testFile, world_size, world_rank, optArgs.isBulk());
        dataSet.distributedLoad(Xtrain, ytrain, Xtest, ytest);
//        dataPerMachine = dataSet.getDataPerMachine();
//        if(world_rank==0) {
//            cout << "From Main : " << "Data Per Machine : " << dataPerMachine << endl;
//            Util util;
//            util.print2DMatrix(Xtest, 20, features);
//            printf("\n----------------------------------------\n");
//        }


        PSGD sgd1(0.50, 0.50, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, dataPerMachine,
                  testingSamples, world_size, world_rank);
        double startTime = MPI_Wtime();
        if (optArgs.isIsNormalTime()) {
            double drop_per = optArgs.getDrop_out_per();
            sgd1.adamSGDRandomRingv2(w, drop_per, summarylogfile);
        }

        if (optArgs.isIsEpochTime()) {
            string suffix = getTimeStamp();
            logfile.append(logsourceBase).append("logs/epochlog/").append(datasource).append("/").append(
                    "world_size=").append(to_string(world_size)).append("_iterations=").append(
                    to_string(optArgs.getIterations()));
            logfile.append("_").append(suffix);
            sgd1.adamSGD(w, logfile);
        }


        double endTime = MPI_Wtime();
        if (world_rank == 0) {

            double acc = 0;
            if(!optArgs.isBulk()){
                Predict predict(Xtest, ytest, w, testSet, features);
                acc = predict.predict();
            }else{
                string bulktestfile;
                bulktestfile.append(datasourceBase).append(datasource).append("/bulk/");
                string dir = string(bulktestfile);
                vector<string> files = vector<string>();

                getdir(dir,files);
                int fixedTest = 20000;
                double ytest[fixedTest];
                initializer.initializeWeightsWithArray(testSet, ytest);

                double **Xtest;

                Xtest = new double *[fixedTest];
                for (int i = 0; i < fixedTest; ++i) {
                    Xtest[i] = new double[features];
                }
                double cum_acc = 0;
                for (unsigned int i = 0;i < files.size()-1;i++) {
                    string file="";
                    file.append(bulktestfile).append(files[i]);
                    cout << file << endl;
                    DataSet dataSet1(features, fixedTest,file);
                    dataSet1.loadTestData(Xtest, ytest);
                    Predict predict(Xtest, ytest, w, testSet, features);
                    acc = predict.predict();
                    cout << "Test " << i << ", Accuracy : " << acc << endl;
                    cum_acc += acc;
                }
                acc = cum_acc / double(files.size());
            }
            cout << "Training Time : " << (endTime - startTime) << endl;
            cout << "Testing Accuracy : " << acc << "%" << endl;
            summary(summarylogfile, world_size, acc, (endTime - startTime), datasource);
        }

        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < dataPerMachine; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;
    }
    MPI_Finalize();
}


void parallel(OptArgs optArgs) {
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    optArgs.toString();
    ResourceManager resourceManager;
    resourceManager.loadDataSourcePath();
    if (optArgs.isIsSplit()) {
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string sourceFile;
        sourceFile.append(datasourceBase).append(datasource).append(trainFileName);
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();
        double ratio = optArgs.getRatio();
        DataSet dataSet(sourceFile, features, trainingSamples, optArgs.isIsSplit(), ratio);
        dataSet.load();

        double **Xtrain = dataSet.getXtrain();
        double *ytrain = dataSet.getYtrain();

        double **Xtest = dataSet.getXtest();
        double *ytest = dataSet.getYtest();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples * ratio;
        int testSet = totalSamples - trainSet;
        Util util;
        util.print2DMatrix(Xtrain, trainSet, features);
        printf("\n----------------------------------------\n");
        util.print2DMatrix(Xtest, testSet, features);
        //PSGD sgd1(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, trainingSamples, testingSamples, world_size, world_rank);
        //double startTime = MPI_Wtime();
        //sgd1.adamSGD();
        //double endTime = MPI_Wtime();
//        if(world_rank ==0) {
//            cout << "Training Time : " << (endTime - startTime) << endl;
//        }



    } else {
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string trainFilePath;
        trainFilePath.append(datasourceBase).append(datasource).append(trainFileName);
        string testFilePath;
        testFilePath.append(datasourceBase).append(datasource).append(testFileName);
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();

        DataSet dataset(features, trainingSamples, testingSamples, trainFilePath, testFilePath);
        dataset.load();
        double **Xtrain = dataset.getXtrain();
        double *ytrain = dataset.getYtrain();

        double **Xtest = dataset.getXtest();
        double *ytest = dataset.getYtest();

//        if(world_rank==0) {
//            Util util;
//            util.print2DMatrix(Xtrain, trainingSamples, features);
//            printf("\n----------------------------------------\n");
//            util.print2DMatrix(Xtest, testingSamples, features);
//        }

//        clock_t begin = clock();

        PSGD sgd1(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, trainingSamples,
                  testingSamples, world_size, world_rank);
        double startTime = MPI_Wtime();
        sgd1.adamSGD();
        double endTime = MPI_Wtime();
        if (world_rank == 0) {
            cout << "Training Time : " << (endTime - startTime) << endl;
        }

//        //sgd1.sgd();
//        clock_t end = clock();
//        double elapsed_secs = double((end - begin) / double(CLOCKS_PER_SEC));
//        printf("Training Samples : % d \n", trainingSamples);
//        printf("Testing Samples : % d \n", testingSamples);
//        printf("Training time %f s \n", elapsed_secs);
//        double* wFinal = sgd1.getWFinal();
//        util.print1DMatrix(wFinal, features);
//        Predict predict(Xtest, ytest, wFinal , testingSamples, features);
//        double acc = predict.predict();
//        cout << "Testing Accuracy : " << acc << "%" << endl;
    }
    MPI_Finalize();
}

void train(OptArgs optArgs) {
    optArgs.toString();
    ResourceManager resourceManager;
    resourceManager.loadDataSourcePath();
    if (optArgs.isIsSplit()) {
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string sourceFile;
        sourceFile.append(datasourceBase).append(datasource).append(trainFileName);
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        double ratio = optArgs.getRatio();
        DataSet dataSet(sourceFile, features, trainingSamples, optArgs.isIsSplit(), ratio);
        dataSet.load();

        double **Xtrain = dataSet.getXtrain();
        double *ytrain = dataSet.getYtrain();

        double **Xtest = dataSet.getXtest();
        double *ytest = dataSet.getYtest();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples * ratio;
        int testSet = totalSamples - trainSet;
        Util util;
        //util.print2DMatrix(Xtrain, trainSet, features);
        printf("\n----------------------------------------\n");
        //util.print2DMatrix(Xtest, testSet, features);
        clock_t begin = clock();
        SGD sgd1(Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, trainSet, testSet);
        //SGD sgd2(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, trainSet);
        sgd1.sgd();
        //sgd1.sgd();
        clock_t end = clock();
        double elapsed_secs = double((end - begin) / double(CLOCKS_PER_SEC));
        printf("Training Samples : % d \n", trainSet);
        printf("Testing Samples : % d \n", testSet);
        printf("Training time %f s \n", elapsed_secs);
        //double* wFinal = sgd2.getWFinal();
        //util.print1DMatrix(wFinal, features);
//        double  wFinalTest [22] = {1.058225086609490377e-02,
//                2.280953604657339449e-03,
//                -1.943556791991714111e-05,
//                -1.077843995538567932e-02,
//                -8.031186202692977907e-03,
//                5.604057416423767826e-04,
//                -3.297580529278739542e-02,
//                -7.734271973922541600e-04,
//                -5.926790741002434248e-03,
//                -7.005303267160217610e-03,
//                5.329900037406029023e-01,
//                -5.146574347588905418e+00,
//                1.647242436140160526e-01,
//                8.668693932380384937e-02,
//                6.062124153509106800e-02,
//                8.558154902527673191e-02,
//                -2.200800258968939604e-01,
//                -3.348020131636486596e-01,
//                -1.974553008397789966e-01,
//                1.495450499343256301e-01,
//                1.682656341270570011e-01,
//                1.148141532462719216e-01};
//        Predict predict(Xtest, ytest, wFinalTest , testSet, features);
//        double acc = predict.predict();
//        cout << "Testing Accuracy : " << acc << "%" << endl;


    } else {
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string trainFilePath;
        trainFilePath.append(datasourceBase).append(datasource).append(trainFileName);
        string testFilePath;
        testFilePath.append(datasourceBase).append(datasource).append(testFileName);
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();

        DataSet dataset(features, trainingSamples, testingSamples, trainFilePath, testFilePath);
        dataset.load();
        double **Xtrain = dataset.getXtrain();
        double *ytrain = dataset.getYtrain();

        double **Xtest = dataset.getXtest();
        double *ytest = dataset.getYtest();
        Util util;
//        util.print2DMatrix(Xtrain, trainingSamples, features);
        printf("\n----------------------------------------\n");
//        util.print2DMatrix(Xtest, testingSamples, features);
        clock_t begin = clock();
        SGD sgd1(Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, trainingSamples,
                 testingSamples);
        //SGD sgd2(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, trainingSamples);
        //sgd2.adamSGD();
        sgd1.sgd();
        clock_t end = clock();
        double elapsed_secs = double((end - begin) / double(CLOCKS_PER_SEC));
        printf("Training Samples : % d \n", trainingSamples);
        printf("Testing Samples : % d \n", testingSamples);
        printf("Training time %f s \n", elapsed_secs);
        //double* wFinal = sgd2.getWFinal();
        //util.print1DMatrix(wFinal, features);
//        double  wFinalTest [22] = {1.058225086609490377e-02,
//                                   2.280953604657339449e-03,
//                                   -1.943556791991714111e-05,
//                                   -1.077843995538567932e-02,
//                                   -8.031186202692977907e-03,
//                                   5.604057416423767826e-04,
//                                   -3.297580529278739542e-02,
//                                   -7.734271973922541600e-04,
//                                   -5.926790741002434248e-03,
//                                   -7.005303267160217610e-03,
//                                   5.329900037406029023e-01,
//                                   -5.146574347588905418e+00,
//                                   1.647242436140160526e-01,
//                                   8.668693932380384937e-02,
//                                   6.062124153509106800e-02,
//                                   8.558154902527673191e-02,
//                                   -2.200800258968939604e-01,
//                                   -3.348020131636486596e-01,
//                                   -1.974553008397789966e-01,
//                                   1.495450499343256301e-01,
//                                   1.682656341270570011e-01,
//                                   1.148141532462719216e-01};
        //Predict predict(Xtest, ytest, wFinal , testingSamples, features);
        //double acc = predict.predict();
        //cout << "Testing Accuracy : " << acc << "%" << endl;
    }
}


void trainSequential(OptArgs optArgs) {
    optArgs.toString();
    ResourceManager resourceManager;
    resourceManager.loadDataSourcePath();
    resourceManager.loadEpochSummaryPath();
    resourceManager.loadLogSourcePath();
    resourceManager.loadSummaryPath();
    resourceManager.loadWeightSummaryPath();
    string exptype = "adam";
    if (optArgs.isIsSplit()) {
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string datasource = optArgs.getDataset();

        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";

        string epochlogfile = resourceManager.getEpochlogSummaryBasePath();
        epochlogfile.append("sequential/").append(exptype).append("/").append(datasource).append("/").append(getTimeStamp()).append("_").append("sequential_cross_validation_accuracy.csv");

        string summarylogfile = resourceManager.getLogSummaryBasePath();
        summarylogfile.append("sequential/").append(exptype).append("/").append(datasource).append("/").append(getTimeStamp()).append("_").append("_sequential_summary_log.csv");

        string sourceFile;
        sourceFile.append(datasourceBase).append(datasource).append(trainFileName);
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples * ratio;
        int testSet = totalSamples - trainSet;
        Initializer initializer;

        double ytrain[trainSet];
        initializer.initializeWeightsWithArray(trainSet, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[trainSet];
        for (int i = 0; i < trainSet; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        Util util;

        DataSet dataSet(sourceFile, features, trainingSamples, optArgs.isIsSplit(), ratio);
        dataSet.load(Xtrain, ytrain, Xtest, ytest);

        util.print2DMatrix(Xtrain, 2, features);
        printf("\n----------------------------------------\n");
        util.print2DMatrix(Xtest, 2, features);
        clock_t begin = clock();
        double *w = new double[features];
        //SGD sgd1(Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, trainSet, testSet);
        SGD sgd2(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, trainSet);
        SGD sgd3(0.5,0.5, Xtrain, ytrain, w, optArgs.getAlpha(), optArgs.getIterations(), features, trainSet, testSet, Xtest, ytest);
        sgd3.adamSGD(w,summarylogfile, epochlogfile);
        //sgd1.sgd();
        clock_t end = clock();
        double elapsed_secs = double((end - begin) / double(CLOCKS_PER_SEC));
        printf("Training Samples : % d \n", trainSet);
        printf("Testing Samples : % d \n", testSet);
        printf("Training time %f s \n", elapsed_secs);



        Predict predict(Xtest, ytest, w , testSet, features);
        double acc = predict.predict();
        cout << "Testing Accuracy : " << acc << "%" << endl;
        util.summary(summarylogfile, 1, acc, elapsed_secs);
        for (int i = 0; i < trainSet; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < testSet; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;


    } else {
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string trainFilePath;
        trainFilePath.append(datasourceBase).append(datasource).append(trainFileName);
        string testFilePath;
        testFilePath.append(datasourceBase).append(datasource).append(testFileName);
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();

        DataSet dataset(features, trainingSamples, testingSamples, trainFilePath, testFilePath);
        dataset.load();
        double **Xtrain = dataset.getXtrain();
        double *ytrain = dataset.getYtrain();

        double **Xtest = dataset.getXtest();
        double *ytest = dataset.getYtest();
        Util util;
//        util.print2DMatrix(Xtrain, trainingSamples, features);
        printf("\n----------------------------------------\n");
//        util.print2DMatrix(Xtest, testingSamples, features);
        clock_t begin = clock();
        double *w = new double[features];
        //SGD sgd1(Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, trainingSamples, testingSamples);
        SGD sgd2(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, trainingSamples);
        sgd2.adamSGD(w);
        //sgd1.adamSGD();
        clock_t end = clock();
        double elapsed_secs = double((end - begin) / double(CLOCKS_PER_SEC));
        printf("Training Samples : % d \n", trainingSamples);
        printf("Testing Samples : % d \n", testingSamples);
        printf("Training time %f s \n", elapsed_secs);

        //Predict predict(Xtest, ytest, wFinal , testingSamples, features);
        //double acc = predict.predict();
        //cout << "Testing Accuracy : " << acc << "%" << endl;
        delete[] Xtrain, Xtest, ytrain, ytest, w;
    }

}

void sequentialPegasos(OptArgs optArgs) {
    optArgs.toString();
    ResourceManager resourceManager;
    resourceManager.loadDataSourcePath();
    resourceManager.loadEpochSummaryPath();
    resourceManager.loadLogSourcePath();
    resourceManager.loadSummaryPath();
    resourceManager.loadWeightSummaryPath();
    string epochlogfile = resourceManager.getEpochlogSummaryBasePath();
    string datasourceBase = resourceManager.getDataSourceBasePath();
    string datasource = optArgs.getDataset();

    epochlogfile.append("sequential_pegasos/").append(datasource).append("/").append(getTimeStamp()).append("_").append("sequential_pegasos_cross_validation_accuracy.csv");

    string summarylogfile = resourceManager.getLogSummaryBasePath();
    summarylogfile.append("sequential_pegasos/").append(datasource).append("/").append(getTimeStamp()).append("_").append("_sequential_pegasos_summary_log.csv");

    if (optArgs.isIsSplit()) {
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";

        string sourceFile;
        sourceFile.append(datasourceBase).append(datasource).append(trainFileName);
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples * ratio;
        int testSet = totalSamples - trainSet;
        Initializer initializer;

        double ytrain[trainSet];
        initializer.initializeWeightsWithArray(trainSet, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[trainSet];
        for (int i = 0; i < trainSet; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        Util util;

        DataSet dataSet(sourceFile, features, trainingSamples, optArgs.isIsSplit(), ratio);
        dataSet.load(Xtrain, ytrain, Xtest, ytest);

        util.print2DMatrix(Xtrain, 2, features);
        printf("\n----------------------------------------\n");
        util.print2DMatrix(Xtest, 2, features);

        double *w = new double[features];
        //SGD sgd1(Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, trainSet, testSet);
        SGD sgd2(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, trainSet);
        SGD sgd3(0.5,0.5, Xtrain, ytrain, w, optArgs.getAlpha(), optArgs.getIterations(), features, trainSet, testSet, Xtest, ytest);
        sgd3.setError_threshold(optArgs.getError_threshold());
        clock_t begin = clock();
        if(optArgs.isPegasosSeqNoTime()) {
            sgd3.pegasosSgdNoTiming(w, summarylogfile, epochlogfile);
        } else {
            sgd3.pegasosSgd(w,summarylogfile, epochlogfile);
        }

        //sgd1.sgd();
        clock_t end = clock();
        double elapsed_secs = double((end - begin) / double(CLOCKS_PER_SEC)) - (sgd3.getTotalPredictionTime());
        printf("Training Samples : % d \n", trainSet);
        printf("Testing Samples : % d \n", testSet);
        printf("Training time %f s \n", elapsed_secs);


        Predict predict(Xtest, ytest, w , testSet, features);
        double acc = predict.predict();
        cout << "Testing Accuracy : " << acc << "%" << endl;
        util.summary(summarylogfile, 1, acc, elapsed_secs, optArgs.getAlpha(), optArgs.getError_threshold());
        for (int i = 0; i < trainSet; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < testSet; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;


    } else {
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string trainFilePath;
        trainFilePath.append(datasourceBase).append(datasource).append(trainFileName);
        string testFilePath;
        testFilePath.append(datasourceBase).append(datasource).append(testFileName);
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();

        DataSet dataset(features, trainingSamples, testingSamples, trainFilePath, testFilePath);
        dataset.load();
        double **Xtrain = dataset.getXtrain();
        double *ytrain = dataset.getYtrain();

        double **Xtest = dataset.getXtest();
        double *ytest = dataset.getYtest();
        Util util;
//        util.print2DMatrix(Xtrain, trainingSamples, features);
        printf("\n----------------------------------------\n");
//        util.print2DMatrix(Xtest, testingSamples, features);
        clock_t begin = clock();
        double *w = new double[features];
        //SGD sgd1(Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, trainingSamples, testingSamples);
        SGD sgd2(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, trainingSamples);
        if(optArgs.isPegasosSeqNoTime()) {
            sgd2.pegasosSgdNoTiming(w, summarylogfile, epochlogfile);
        } else {
            sgd2.pegasosSgd(w, summarylogfile, epochlogfile);
        }

        //sgd1.adamSGD();
        clock_t end = clock();
        double elapsed_secs = double((end - begin) / double(CLOCKS_PER_SEC)) - (sgd2.getTotalPredictionTime());
        printf("Training Samples : % d \n", trainingSamples);
        printf("Testing Samples : % d \n", testingSamples);
        printf("Training time %f s \n", elapsed_secs);

        Predict predict(Xtest, ytest, w , testingSamples, features);
        double acc = predict.predict();
        cout << "Testing Accuracy : " << acc << "%" << endl;
        util.summary(summarylogfile, 1, acc, elapsed_secs);
        //Predict predict(Xtest, ytest, wFinal , testingSamples, features);
        //double acc = predict.predict();
        //cout << "Testing Accuracy : " << acc << "%" << endl;
        for (int i = 0; i < trainingSamples; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < testingSamples; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;
    }

}

void sequentialPegasosBatchV1(OptArgs optArgs, int comm_gap) {
    optArgs.toString();
    ResourceManager resourceManager;
    resourceManager.loadDataSourcePath();
    resourceManager.loadEpochSummaryPath();
    resourceManager.loadLogSourcePath();
    resourceManager.loadSummaryPath();
    resourceManager.loadWeightSummaryPath();
    string epochlogfile = resourceManager.getEpochlogSummaryBasePath();
    string datasourceBase = resourceManager.getDataSourceBasePath();
    string datasource = optArgs.getDataset();

    epochlogfile.append("sequential_pegasos/").append(datasource).append("/").append(getTimeStamp()).append("_").append("sequential_batch_pegasos_cross_validation_accuracy.csv");

    string summarylogfile = resourceManager.getLogSummaryBasePath();
    summarylogfile.append("sequential_pegasos/").append(datasource).append("/").append(getTimeStamp()).append("_").append("_sequential__batch_pegasos_summary_log.csv");

    if (optArgs.isIsSplit()) {
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";

        string sourceFile;
        sourceFile.append(datasourceBase).append(datasource).append(trainFileName);
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        double ratio = optArgs.getRatio();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples * ratio;
        int testSet = totalSamples - trainSet;
        Initializer initializer;

        double ytrain[trainSet];
        initializer.initializeWeightsWithArray(trainSet, ytrain);


        double ytest[testSet];
        initializer.initializeWeightsWithArray(testSet, ytest);

        double **Xtrain;
        double **Xtest;
        Xtrain = new double *[trainSet];
        for (int i = 0; i < trainSet; ++i) {
            Xtrain[i] = new double[features];
        }

        Xtest = new double *[testSet];
        for (int i = 0; i < testSet; ++i) {
            Xtest[i] = new double[features];
        }


        Util util;

        DataSet dataSet(sourceFile, features, trainingSamples, optArgs.isIsSplit(), ratio);
        dataSet.load(Xtrain, ytrain, Xtest, ytest);

        util.print2DMatrix(Xtrain, 2, features);
        printf("\n----------------------------------------\n");
        util.print2DMatrix(Xtest, 2, features);

        double *w = new double[features];
        //SGD sgd1(Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, trainSet, testSet);
        SGD sgd2(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, trainSet);
        SGD sgd3(0.5,0.5, Xtrain, ytrain, w, optArgs.getAlpha(), optArgs.getIterations(), features, trainSet, testSet, Xtest, ytest);
        sgd3.setError_threshold(optArgs.getError_threshold());
        clock_t begin = clock();
        //sgd3.pegasosSgd(w,summarylogfile, epochlogfile);
        sgd3.pegasosBlockSgd(w, summarylogfile, epochlogfile, comm_gap);
        //sgd1.sgd();
        clock_t end = clock();
        double elapsed_secs = double((end - begin) / double(CLOCKS_PER_SEC)) - (sgd3.getTotalPredictionTime());
        printf("Training Samples : % d \n", trainSet);
        printf("Testing Samples : % d \n", testSet);
        printf("Training time %f s \n", elapsed_secs);


        Predict predict(Xtest, ytest, w , testSet, features);
        double acc = predict.predict();
        cout << "Testing Accuracy : " << acc << "%" << endl;
        util.summary(summarylogfile, comm_gap, acc, elapsed_secs, optArgs.getAlpha(), optArgs.getError_threshold());
        for (int i = 0; i < trainSet; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < testSet; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;


    } else {
        string datasourceBase = resourceManager.getDataSourceBasePath();
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string trainFilePath;
        trainFilePath.append(datasourceBase).append(datasource).append(trainFileName);
        string testFilePath;
        testFilePath.append(datasourceBase).append(datasource).append(testFileName);
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();

        DataSet dataset(features, trainingSamples, testingSamples, trainFilePath, testFilePath);
        dataset.load();
        double **Xtrain = dataset.getXtrain();
        double *ytrain = dataset.getYtrain();

        double **Xtest = dataset.getXtest();
        double *ytest = dataset.getYtest();
        Util util;
//        util.print2DMatrix(Xtrain, trainingSamples, features);
        printf("\n----------------------------------------\n");
//        util.print2DMatrix(Xtest, testingSamples, features);
        clock_t begin = clock();
        double *w = new double[features];
        //SGD sgd1(Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, trainingSamples, testingSamples);
        SGD sgd2(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, trainingSamples);
        //sgd2.pegasosSgd(w, summarylogfile, epochlogfile);
        sgd2.pegasosBlockSgd(w, summarylogfile, epochlogfile, comm_gap);
        //sgd1.adamSGD();
        clock_t end = clock();
        double elapsed_secs = double((end - begin) / double(CLOCKS_PER_SEC)) - (sgd2.getTotalPredictionTime());
        printf("Training Samples : % d \n", trainingSamples);
        printf("Testing Samples : % d \n", testingSamples);
        printf("Training time %f s \n", elapsed_secs);

        Predict predict(Xtest, ytest, w , testingSamples, features);
        double acc = predict.predict();
        cout << "Testing Accuracy : " << acc << "%" << endl;
        util.summary(summarylogfile, 1, acc, elapsed_secs);
        //Predict predict(Xtest, ytest, wFinal , testingSamples, features);
        //double acc = predict.predict();
        //cout << "Testing Accuracy : " << acc << "%" << endl;
        for (int i = 0; i < trainingSamples; ++i) {
            delete[] Xtrain[i];
        }
        for (int i = 0; i < testingSamples; ++i) {
            delete[] Xtest[i];
        }
        delete[] Xtrain;
        delete[] Xtest;
        delete[] w;
    }
}

void sgd() {
    string datasourceBase = "/home/vibhatha/data/svm/";
    string datasource = "heart";
    string trainFileName = "/training.csv";
    string testFileName = "/testing.csv";
    string sourceFile;
    sourceFile.append(datasourceBase).append(datasource).append(trainFileName);
    int features = 13;
    int trainingSamples = 216;
    int testingSamples = 64;

    DataSet dataset(sourceFile, features, trainingSamples, testingSamples);
    dataset.load();
    double **Xtrain = dataset.getXtrain();
    double *ytrain = dataset.getYtrain();


    Util util;
    //util.print2DMatrix(Xtrain, trainingSamples, features);
    clock_t begin = clock();
    SGD sgd1(Xtrain, ytrain, 0.01, 200, features, trainingSamples, testingSamples);
    sgd1.sgd();
    clock_t end = clock();
    double elapsed_secs = double((end - begin) / double(CLOCKS_PER_SEC));
    printf("Time taken %f s", elapsed_secs);

}

void test1() {
    Test test;

    test.test1();
}

void test2() {
    Test test;
    test.test2();
}

void test3() {
    Test test;
    test.test3();
}

void test4() {
    Test test;
    test.test4();
}

void test5() {
    Test test;
    test.test5();
}


void test6() {
    Test test;
    test.test6();
}

void test7() {
    Test test;
    test.test7();
}

void test8(int argc, char **argv) {
    Test test;
    test.test8(argc, argv);
}

string getTimeStamp() {
    string string1;
    time_t t = time(0);   // get time now
    tm *now = localtime(&t);
    string datestring;
    datestring.append(to_string(now->tm_year + 1900)).append("-").append(to_string((now->tm_mon + 1))).append(
            "-").append(to_string(now->tm_mday));
    string timestring;
    timestring.append(to_string(now->tm_hour)).append(":").append(to_string(now->tm_min)).append(":").append(
            to_string(now->tm_sec));
    string1.append(datestring).append("__").append(timestring);
    return string1;
}

void summary(string logfile, int world_size, double acc, double time, string datasource) {

    ofstream myfile(logfile, ios::out | ios::app);
    string timestamp = getTimeStamp();
    if (myfile.is_open()) {

        myfile << datasource << "," << world_size << "," << time << "," << acc << "," << timestamp << "\n";

        myfile.close();
    }
}

void summary(string logfile, int world_size, double acc, double time, string datasource, double alpha) {

    ofstream myfile(logfile, ios::out | ios::app);
    string timestamp = getTimeStamp();
    if (myfile.is_open()) {

        myfile << datasource << "," << world_size << "," << time << "," << acc << "," << alpha << "," << timestamp << "\n";

        myfile.close();
    }
}

void summary(string logfile, int world_size, double acc, double time, string datasource, double alpha, double error_threshold) {

    ofstream myfile(logfile, ios::out | ios::app);
    string timestamp = getTimeStamp();
    if (myfile.is_open()) {

        myfile << datasource << "," << world_size << "," << time << "," << acc << "," << alpha << "," << timestamp << "," << error_threshold << "\n";

        myfile.close();
    }
}

int getdir (string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}