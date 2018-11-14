//
// Created by vibhatha on 11/5/18.
//

#include "Matrix.h"
#include <cmath>
#include <iostream>

using namespace std;


Matrix::Matrix(int features_) {
    features = features_;
}

double* Matrix::add(double* a, double *b) {
    double* res = new double[features];
    for (int i = 0; i < features; ++i) {
        res[i] = a[i] + b[i];
    }
    a = res;
    delete [] res;
    return a;
}

double* Matrix::subtract(double* a, double *b) {
    double* res = new double[features];
    for (int i = 0; i < features; ++i) {
        res[i] = a[i] - b[i];
    }
    a = res;
    delete [] res;
    return a;
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
    a = res;
    delete [] res;
    return a;
}

double* Matrix::divide(double *a, double *b) {
    double* res = new double[features];
    for (int i = 0; i < features; ++i) {
        res[i] = a[i] / b[i];
    }
    a = res;
    delete [] res;
    return a;
}

double* Matrix::sqrt(double* a) {
    double* res = new double[features];
    for (int i = 0; i < features; ++i) {
        double d = a[i];
        res[i] = std::sqrt(d);
    }
    a = res;
    delete [] res;
    return a;
}

double* Matrix::scalarAddition(double *a, double c) {
    double* res = new double[features];
    for (int i = 0; i < features; ++i) {
        res[i] = a[i] + c;
    }
    a = res;
    delete [] res;
    return a;
}

double* Matrix::put(double *a, double *b) {
    for (int i = 0; i < features; ++i) {
        a[i] = b[i] ;
    }
    return a;
}