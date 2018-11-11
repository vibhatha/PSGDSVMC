//
// Created by vibhatha on 11/11/18.
//

#ifndef PSGDC_PSGD_H
#define PSGDC_PSGD_H


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

    double *getW() const;

    double *getWFinal() const;

    void setWFinal(double *wFinal);

};


#endif //PSGDC_PSGD_H
