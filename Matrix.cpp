#include "Matrix.h"
#include <cmath>
#include <iostream>

using namespace std;


double* Matrix::add(double* a, double *b) {
    for (int i = 0; i < features; ++i) {
        res[i] = a[i] + b[i];
    }
    return res;
}

double* Matrix::subtract(double* a, double *b) {
    for (int i = 0; i < features; ++i) {
        res[i] = a[i] - b[i];
    }
    return res;
}

double* Matrix::scalarMultiply(double* a, double c) {
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
    for (int i = 0; i < features; ++i) {
        res[i] = a[i] *b[i];
    }
    return res;
}

double* Matrix::divide(double *a, double *b) {
    for (int i = 0; i < features; ++i) {
        res[i] = a[i] / b[i];
    }
    return res;
}

double* Matrix::sqrt(double* a) {
    for (int i = 0; i < features; ++i) {
        double d = a[i];
        res[i] = std::sqrt(d);
    }
    return res;
}

double* Matrix::scalarAddition(double *a, double c) {
    double* res = new double[features];
    for (int i = 0; i < features; ++i) {
        res[i] = a[i] + c;
    }
    return res;
}

double* Matrix::put(double *a, double *b) {
    for (int i = 0; i < features; ++i) {
        a[i] = b[i] ;
    }
    return a;
}

const double *Matrix::getRes() const {
    return res;
}

Matrix::Matrix(const int features) : features(features) {
    res = new double[features];
}
