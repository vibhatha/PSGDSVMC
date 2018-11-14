//
// Created by vibhatha on 11/5/18.
//

#ifndef PSGDC_MATRIX_H
#define PSGDC_MATRIX_H


#include "Util.h"

class Matrix {

private:
    int features;

    Util util;

public:
    Matrix(int features);

    double* add(double* a, double* b, double* res);
    double* subtract(double* a, double* b, double* res);
    double dot(double* a, double* b);
    double* scalarMultiply(double* a, double c, double* res);
    double* scalarAddition(double* a, double c, double* res);
    double* inner(double* a, double* b, double* res);
    double* divide(double* a, double* b, double* res);
    double* sqrt(double* a, double* res);
    double* put(double* a, double* b, double* res);



};


#endif //PSGDC_MATRIX_H
