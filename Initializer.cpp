//
// Created by vibhatha on 11/5/18.
//


#include <iostream>
#include <random>
#include "Initializer.h"

using namespace std;

double* Initializer::initialWeights(int features) {
    const int range_from  = 0.0;
    const int range_to    = 1.0;
    random_device                  rand_dev;
    wInit = new double[features];
    for (int i = 0; i < features; ++i) {
        mt19937                        generator(rand_dev());

        uniform_real_distribution<double>  distr(range_from, range_to);
        double val =  distr(generator);
        wInit[i] = val;
    }
    return wInit;
}

void Initializer::initialWeights(int features, double* w) {
    const int range_from  = 0.0;
    const int range_to    = 1.0;
    random_device                  rand_dev;

    for (int i = 0; i < features; ++i) {
        mt19937                        generator(rand_dev());

        uniform_real_distribution<double>  distr(range_from, range_to);
        double val =  distr(generator);
        w[i] = val;
    }

}

double* Initializer::zeroWeights(int features) {
    wInit = new double[features];
    for (int i = 0; i < features; ++i) {
        wInit[i] = 0.0;
    }
    return wInit;
}

double* Initializer::initializeWeightsWithArray(int features, double *a) {
    for (int i = 0; i < features; ++i) {
        a[i] = 0.0;
    }
    return a;
}


double** Initializer::initalizeMatrix(int rows, int columns, double **b) {
    double** temp  = new double*[rows];
    for (int i = 0; i < rows; ++i) {
        temp[i] = new double[columns];
        for (int j = 0; j < columns; ++j) {
            temp[i][j] = 0;
        }
    }
    b = temp;
    delete [] temp;
}


