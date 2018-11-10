//
// Created by vibhatha on 11/5/18.
//

#ifndef PSGDC_SGD_H
#define PSGDC_SGD_H


class SGD {

private:
    double** X;
    double* y;
    double* w;
    double* wInit;
    double alpha;
    int iterations;
    int features;
    int trainingSamples;
    int testingSamples;

public:

    SGD(double** X, double* y, double alpha, int iterations);
    SGD(double** X, double* y, double alpha, int iterations, int features, int trainingSamples,int testingSamples);
    void sgd();

    double *getW() const;

};


#endif //PSGDC_SGD_H
