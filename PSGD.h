//
// Created by vibhatha on 11/11/18.
//

#ifndef PSGDC_PSGD_H
#define PSGDC_PSGD_H

#include <iostream>
#include <vector>
#include "armadillo"

using namespace std;

class PSGD {
private:
    double beta1=0.93;
    double beta2=0.999;
    double** X;
    double* y;
    double* w;
    double* wInit;
    double alpha=0.01;
    int iterations;
    int features;
    int trainingSamples;
    int testingSamples;
    double* wFinal;
    int world_size;
    int world_rank;
    double compute_time=0;
    double communication_time=0;
    vector<double> compute_time_of_ranks;
    vector<double> communication_time_of_ranks;
    double** Xtest;
    double* ytest;
    double totalPredictionTime;
    double error_threshold;
    int effective_epochs;
    double resultant_minimum_cost=0;
    double resultant_final_cross_accuracy=0;

public:
    PSGD(double** X, double* y, double alpha, int iterations);
    PSGD(double** X, double* y, double alpha, int iterations, int features, int trainingSamples,int testingSamples);

    PSGD(double beta1, double beta2, double **X, double *y, double alpha, int iterations, int features,
        int trainingSamples);

    PSGD(double beta1, double beta2, double alpha, int iterations, int features, int trainingSamples,
        int testingSamples);

    PSGD(double beta1, double beta2, double **X, double *y, double alpha, int iterations, int features, int world_size,
         int world_rank);

    PSGD(double beta1, double beta2, double **X, double *y, double alpha, int iterations, int features,
         int trainingSamples, int testingSamples, int world_size, int world_rank);

    PSGD(double beta1, double beta2, double **X, double *y, double alpha, int iterations, int features,
         int trainingSamples, int testingSamples, int world_size, int world_rank, double **Xtest, double *ytest);


    void sgd();
    void adamSGDSeq(double* w);
    void adamSGDSeq(double* w, string logfile);
    void adamSGD();
    void adamSGD(double* w);
    void adamSGD(double* w, string logfile);
    void writeLog(string logfile, int iterations, int samples, double** compt, double** commt);
    void writeVectorLog(string logfile, int iterations, int samples, vector<double> compt, vector<double> commt);
    void writeLog(string logfile, int iterations, int samples, double** compt, double commt);
    void writeLog(string logfile, int iterations, int samples, double** compt, double* commt);
    void adamSGDBatchv1(double* w);
    void adamSGDBatchv1(double* w, string logfile);
    void adamSGDBatchv2(double* w, int comm_gap);
    void adamSGDBatchv2(double* w, int comm_gap, string logfile);
    void adamSGDBatchv2(double* w, int comm_gap, string logfile, string epochlogfile);
    void sgdBatchv2(double* w, int comm_gap, string logfile, string epochlogfile);
    void adamSGDRotationv1(double* w);
    void adamSGDRandomRingv1(double* w, double dropout_per, string logfile);
    void adamSGDRandomRingv2(double* w, double dropout_per, string logfile);
    void adamSGDFullBatchv1(double *w, string logfile, string epochlogfile);
    void adamSGDFullBatchv2(double *w, string epochlogfile);
    void sgdFullBatchv1(double *w, string logfile, string epochlogfile);
    void sgdFullBatchv2(double *w, string epochlogfile);
    void pegasosSGDFullBatchv1(double *w, string epochlogfile);
    void pegasosSGDFullBatchv2(double *w, string epochlogfile);
    void pegasosSGDFullBatchv3(double *w, string epochlogfile);
    void pegasosSGDBatchv2(double *w, int comm_gap, string summarylogfile, string epochlogfile, string weightfile);
    void pegasosSGDBatchv2t1(double *w, int comm_gap, int threads, string summarylogfile, string epochlogfile, string weightfile);
    void blassTest();
    void nonBlassTest();


    double *getW() const;

    double *getWFinal() const;

    void setWFinal(double *wFinal);

    double getCompute_time() const;

    void setCompute_time(double compute_time);

    double getCommunication_time() const;

    void setCommunication_time(double communication_time);

    const vector<double> &getCompute_time_of_ranks() const;

    const vector<double> &getCommunication_time_of_ranks() const;

    bool isPresent(int* arr, int new_rank, int size);

    bool isPossibleRanks(int* arr, int size);

    void generateRandomRanks(int *arr, double per);

    bool isIncluded(int* active_ranks, int my_rank, int size);

    int getRankIndex(int* active_ranks, int my_rank, int size);

    double getTotalPredictionTime() const;

    void setTotalPredictionTime(double totalPredictionTime);

    double getError_threshold() const;

    void setError_threshold(double error_threshold);

    int getEffective_epochs() const;

    void setEffective_epochs(int effective_epochs);

    double getResultant_minimum_cost() const;

    void setResultant_minimum_cost(double resultant_minimum_cost);

    double getResultant_final_cross_accuracy() const;

    void setResultant_final_cross_accuracy(double resultant_final_cross_accuracy);

};


#endif //PSGDC_PSGD_H
