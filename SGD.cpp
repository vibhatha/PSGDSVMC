//
// Created by vibhatha on 11/5/18.
//

#include <iostream>
#include "SGD.h"
#include "Initializer.h"

using namespace std;

SGD::SGD(double **Xn, double* yn, double alphan, int itrN) {
    X = Xn;
    y = yn;
    alpha = alphan;
    iterations = itrN;
}

SGD::SGD(double **Xn, double *yn, double alphan, int itrN, int features_, int trainingSamples_, int testingSamples_) {
    X = Xn;
    y = yn;
    alpha = alphan;
    iterations = itrN;
    features = features_;
    trainingSamples = trainingSamples_;
    testingSamples = testingSamples_;
}

void SGD::sgd() {
    Initializer initializer;
    wInit = initializer.initialWeights(features);

//    for (int i = 0; i < iterations; ++i) {
//        for (int j = 0; j < trainingSamples; ++j) {
//            double* xi = X[j];
//            double yi = y[j];
//        }
//    }
}
