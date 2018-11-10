//
// Created by vibhatha on 11/10/18.
//

#include "Predict.h"
#include "Matrix.h"
#include <iostream>

using namespace std;


double Predict::predict() {
    double accuracy = 0;
    double totalCorrect = 0;
    Matrix matrix(features);
    for (int i = 0; i < testingSamples; ++i) {
        double pred = 0;
        double d = matrix.dot(w, X[i]);
        if(d>=0) {
            pred = 1;
        }
        if(d<0) {
            pred = -1;
        }
        if(y[i] == pred) {
            totalCorrect++;
        }
        //cout << i << " : " << pred << "/" << y[i] << endl;

    }
    accuracy = (totalCorrect / testingSamples) * 100.0;

    return accuracy;
}

Predict::Predict(double **X, double *y, double *w, int testingSamples, int features) : X(X), y(y), w(w),
                                                                                       testingSamples(testingSamples),
                                                                                       features(features) {}
