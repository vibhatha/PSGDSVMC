//
// Created by vibhatha on 11/5/18.
//

#ifndef PSGDC_INITIALIZER_H
#define PSGDC_INITIALIZER_H


class Initializer {
private:
    double* wInit;

public:
    double* initialWeights(int features);
    double* zeroWeights(int features);
    double* initializeWeightsWithArray(int features, double a []);
    double** initalizeMatrix(int rows, int columns, double** b );
};


#endif //PSGDC_INITIALIZER_H
