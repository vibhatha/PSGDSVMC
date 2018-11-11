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

double* Initializer::zeroWeights(int features) {
    wInit = new double[features];
    for (int i = 0; i < features; ++i) {
        wInit[i] = 0.0;
    }
    return wInit;
}

