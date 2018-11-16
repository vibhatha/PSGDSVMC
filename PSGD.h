//
// Created by vibhatha on 11/11/18.
//

#ifndef PSGDC_PSGD_H
#define PSGDC_PSGD_H

#include <iostream>
#include <vector>

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

    void sgd();
    void adamSGD();
    void adamSGD(double* w);

    double *getW() const;

    double *getWFinal() const;

    void setWFinal(double *wFinal);

    double getCompute_time() const;

    void setCompute_time(double compute_time);

    double getCommunication_time() const;

    void setCommunication_time(double communication_time);

    const vector<double> &getCompute_time_of_ranks() const;

    const vector<double> &getCommunication_time_of_ranks() const;

};


#endif //PSGDC_PSGD_H
