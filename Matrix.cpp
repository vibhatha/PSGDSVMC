//
// Created by vibhatha on 11/5/18.
//

#include "Matrix.h"


Matrix::Matrix(int features_) {
    features = features_;
}

double* Matrix::add(double* a, double *b) {
    double* res = new double[features];
    for (int i = 0; i < features; ++i) {
        res[i] = a[i] + b[i];
    }
    return res;
}

double* Matrix::subtract(double* a, double *b) {
    double* res = new double[features];
    for (int i = 0; i < features; ++i) {
        res[i] = a[i] - b[i];
    }
    return res;
}

double* Matrix::scalarMultiply(double* a, double c) {
    double* res = new double[features];
    for (int i = 0; i < features; ++i) {
        res[i] = a[i] *c;
    }
    return res;
}

double Matrix::dot(double* a, double* b) {
    double res = 0;
    for (int i = 0; i < features; ++i) {
        res += (a[i]*b[i]) ;
    }
    return res;
}

double* Matrix::inner(double* a, double* b) {
    double* res = new double[features];
    for (int i = 0; i < features; ++i) {
        res[i] = a[i] *b[i];
    }
    return res;
}