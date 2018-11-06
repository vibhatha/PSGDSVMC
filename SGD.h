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
    double alpha;
    int iterations;

public:

    SGD(double** X, double* y, double alpha, int iterations);
    void sgd();

};


#endif //PSGDC_SGD_H
