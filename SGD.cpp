//
// Created by vibhatha on 11/5/18.
//

#include <iostream>
#include "SGD.h"
#include "Initializer.h"
#include "Util.h"
#include "Matrix.h"

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
    Util util;
    cout << "Training Samples : " << trainingSamples << endl;
    util.print1DMatrix(wInit, features);
    Matrix matrix(features);
    w = wInit;
    for (int i = 0; i < iterations; ++i) {
        for (int j = 0; j < trainingSamples; ++j) {
            double* xi = X[j];
            double yi = y[j];
            double yixiw = matrix.dot(xi, w) * yi;
            cout << i << ", " << yixiw << endl;
            if(yixiw<1) {
                double* xiyia = matrix.scalarMultiply(matrix.subtract(w,matrix.scalarMultiply(xi, yi)), alpha);
                w = matrix.subtract(w, xiyia);
            } else {
                double* wa = matrix.scalarMultiply(w,alpha);
                w = matrix.subtract(w,wa);
            }
            //util.print1DMatrix(w,features);
        }
    }
    util.print1DMatrix(w,features);
    printf("Final Weight\n");

}
