//
// Created by vibhatha on 11/5/18.
//

#ifndef PSGDC_SGD_H
#define PSGDC_SGD_H

#include <iostream>

using namespace std;

class SGD {

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
    double** Xtest;
    double* ytest;
    double totalPredictionTime = 0;
    double error_threshold = 0;

public:

    SGD(double** X, double* y, double alpha, int iterations);
    SGD(double** X, double* y, double alpha, int iterations, int features, int trainingSamples,int testingSamples);

    SGD(double beta1, double beta2, double **X, double *y, double alpha, int iterations, int features,
        int trainingSamples);

    SGD(double beta1, double beta2, double alpha, int iterations, int features, int trainingSamples,
        int testingSamples);

    SGD(double beta1, double beta2, double **X, double *y, double *w, double alpha, int iterations, int features,
        int trainingSamples, int testingSamples, double **Xtest, double *ytest);

    void sgd();
    void adamSGD();
    void adamSGD(double* w);
    void adamSGD(double* w, string summarylogfile, string epochlogfile);
    void sgd(double* w, string summarylogfile, string epochlogfile);
    void pegasosSgd(double* w, string summarylogfile, string epochlogfile);

    double *getW() const;

    double *getWFinal() const;

    void setWFinal(double *wFinal);

    double getTotalPredictionTime() const;

    void setTotalPredictionTime(double totalPredictionTime);

    int seed();

    double getError_threshold() const;

    void setError_threshold(double error_threshold);
};


#endif //PSGDC_SGD_H
