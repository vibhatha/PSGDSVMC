//
// Created by vibhatha on 11/14/18.
//

#include "Matrix1.h"
#include <cmath>
#include <iostream>
#include "omp.h"

void Matrix1::scalarMultiply(double *a, double c, double *res) {
    for (int i = 0; i < features; ++i) {
        res[i] = a[i] *c;
    }
}

void Matrix1::parallelScalarMultiply(double *a, double c, double *res) {
    #pragma omp for
    for (int i = 0; i < features; ++i) {
        res[i] = a[i] *c;
    }
}

void Matrix1::subtract(double *a, double *b, double *res) {
    for (int i = 0; i < features; ++i) {
        res[i] = a[i] - b[i];
    }
}

double Matrix1::dot(double *a, double *b) {
    double res = 0;
    for (int i = 0; i < features; ++i) {
        res += (a[i]*b[i]) ;
    }
    return res;
}

void Matrix1::divide(double *a, double *b, double *res) {
    for (int i = 0; i < features; ++i) {
        res[i] = a[i] / b[i];
    }
}

void Matrix1::scalarAddition(double *a, double c, double *res) {
    for (int i = 0; i < features; ++i) {
        res[i] = a[i] + c;
    }
}

void Matrix1::sqrt(double *a, double *res) {
    for (int i = 0; i < features; ++i) {
        double d = a[i];
        res[i] = std::sqrt(d);
    }
}

void Matrix1::add(double *a, double *b, double *res) {
    for (int i = 0; i < features; ++i) {
        res[i] = a[i] + b[i];
    }
}

void Matrix1::inner(double *a, double *b, double *res) {
    for (int i = 0; i < features; ++i) {
        res[i] = a[i] *b[i];
    }
}

Matrix1::Matrix1() {

}

Matrix1::Matrix1(int features) : features(features) {}
