//
// Created by vibhatha on 11/5/18.
//

#ifndef PSGDC_MATRIX_H
#define PSGDC_MATRIX_H


class Matrix {

private:
    int features;
    int samples;
    double* res;

public:
    Matrix(const int features);

    double* add(double* a, double* b);
    double* subtract(double* a, double* b);
    double dot(double* a, double* b);
    double* scalarMultiply(double* a, double c);
    double* scalarAddition(double* a, double c);
    double* inner(double* a, double* b);
    double* divide(double* a, double* b);
    double* sqrt(double* a);
    double* put(double* a, double* b);

    const double *getRes() const;

};


#endif //PSGDC_MATRIX_H
