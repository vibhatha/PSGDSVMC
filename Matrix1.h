//
// Created by vibhatha on 11/14/18.
//

#ifndef PSGDC_MATRIX1_H
#define PSGDC_MATRIX1_H


#include "Util.h"

class Matrix1 {
private:
    int features;

    Util util;

public:
    Matrix1();
    void add(double* a, double* b, double* res);
    void subtract(double* a, double* b, double* res);
    double dot(double* a, double* b);
    void scalarMultiply(double* a, double c, double* res);
    void parallelScalarMultiply(double* a, double c, double* res);
    void scalarAddition(double* a, double c, double* res);
    void inner(double* a, double* b, double* res);
    void divide(double* a, double* b, double* res);
    void sqrt(double* a, double* res);
    void put(double* a, double* b, double* res);

    Matrix1(int features);
};


#endif //PSGDC_MATRIX1_H
