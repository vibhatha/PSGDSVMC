//
// Created by vibhatha on 11/10/18.
//

#include "Predict.h"
#include "Matrix.h"
#include <iostream>
#include "Util.h"

using namespace std;


double Predict::predict() {
    //cout << "Testing Samples : " << testingSamples << endl;
    double accuracy = 0;
    double totalCorrect = 0;
    Matrix matrix(features);
    //Util util;
    //util.print2DMatrix(X, 10, features);
    for (int i = 0; i < testingSamples; ++i) {
        double pred = 0;
        double d = matrix.dot(w, X[i]);
        if(d>=0.0) {
            pred = 1.0;
        }
        if(d<0.0) {
            pred = -1.0;
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
