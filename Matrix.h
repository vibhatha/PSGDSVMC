//
// Created by vibhatha on 11/5/18.
//

#ifndef PSGDC_MATRIX_H
#define PSGDC_MATRIX_H


class Matrix {

private:
    int features;
    int samples;

public:
    Matrix(int features_);
    double* add(double* a, double* b);
    double* subtract(double* a, double* b);
    double dot(double* a, double* b);
    double* scalarMultiply(double* a, double c);
    double* inner(double* a, double* b);


};


#endif //PSGDC_MATRIX_H
