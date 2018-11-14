//
// Created by vibhatha on 11/5/18.
//

#include <iostream>
#include "SGD.h"
#include "Initializer.h"
#include "Util.h"
#include "Matrix.h"
#include "Matrix1.h"
#include <math.h>

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
    double* res = new double[features];
    for (int i = 0; i < iterations; ++i) {
        if(i%10 == 0) {
            cout << "Iteration " << i << "/" << iterations << endl;
        }
        for (int j = 0; j < trainingSamples; ++j) {
            double yixiw = matrix.dot(X[j], w) * y[j];
            //cout << i << ", " << yixiw << endl;
            if(yixiw<1) {
                double* xiyia = matrix.scalarMultiply(matrix.subtract(w,matrix.scalarMultiply(X[j], y[j], res), res), alpha, res);
                w = matrix.subtract(w, xiyia, res);
            } else {
                double* wa = matrix.scalarMultiply(w,alpha, res);
                w = matrix.subtract(w,wa, res);
            }
            //util.print1DMatrix(w,features);
        }
    }
    this->setWFinal(w);
    //util.print1DMatrix(w,features);

    //printf("Final Weight\n");
}

void SGD::adamSGD() {
    Initializer initializer;

    wInit = initializer.initialWeights(features);
    double* v = initializer.zeroWeights(features);
    double* r = initializer.zeroWeights(features);
    double* v1 = initializer.zeroWeights(features);
    double* v2 = initializer.zeroWeights(features);
    double* r1 = initializer.zeroWeights(features);
    double* r2 = initializer.zeroWeights(features);
    double* w1 = initializer.zeroWeights(features);
    double* w2 = initializer.zeroWeights(features);
    double* v_hat = initializer.zeroWeights(features);
    double* r_hat = initializer.zeroWeights(features);
    double* sq_r_hat = initializer.zeroWeights(features);
    double* grad_mul = initializer.zeroWeights(features);
    double* gradient = initializer.zeroWeights(features);
    double* xiyi = initializer.zeroWeights(features);
    double* w_xiyi = initializer.zeroWeights(features);
    double* aw_axiyi = initializer.zeroWeights(features);
    double* w1d = initializer.zeroWeights(features);
    double* aw1 = initializer.zeroWeights(features);
    double epsilon = 0.00000001;
    Util util;
    cout << "Training Samples : " << trainingSamples << endl;
    cout << "Beta 1 :" << beta1 << ", Beta 2 :" << beta2 << endl;
    //util.print1DMatrix(wInit, features);
    //util.print1DMatrix(v, features);
    //util.print1DMatrix(r, features);

    Matrix1 matrix(features);
    double* res = new double[features];
    w = wInit;
    for (int i = 1; i < iterations; ++i) {
        if(i % 10 == 0) {
            //cout << "+++++++++++++++++++++++++++++++++" << endl;
            //util.print1DMatrix(w, features);
            //cout << "+++++++++++++++++++++++++++++++++" << endl;
            //cout << "Iteration " << i << "/" << iterations << endl;
        }
        for (int j = 0; j < trainingSamples; ++j) {

            double yixiw = matrix.dot(X[j], w);
            yixiw = yixiw * y[j];

            double coefficient = 1.0 /(1.0 + double(i));

            if(yixiw<1) {
                matrix.scalarAddition(X[j],y[j],xiyi);
                matrix.subtract(w, xiyi, w_xiyi);
                matrix.scalarMultiply(w_xiyi, alpha, gradient);

            } else {
                matrix.scalarMultiply(w,(1-alpha), gradient);
            }



            matrix.scalarMultiply(v, beta1, v1);
            matrix.scalarMultiply(gradient, (1-beta1), v2);
            matrix.add(v1, v2, v);
            matrix.scalarMultiply(v, (1.0 /(1.0-(pow(beta1,(double)i)))), v_hat);
            matrix.inner(gradient, gradient, grad_mul);
            matrix.scalarMultiply(r, beta2, r1);
            matrix.scalarMultiply(grad_mul, (1-beta2), r2);
            matrix.add(r1, r2, r);
            matrix.scalarMultiply(r, (1.0/(1.0-(pow(beta2,(double)i)))), r_hat);
            //w1 = matrix.sqrt(r_hat);
            //w1 = matrix.scalarAddition(w1,epsilon);
            //w1 = matrix.divide(v_hat,w1);
            matrix.sqrt(r_hat, sq_r_hat);
            matrix.scalarAddition(sq_r_hat, epsilon, w1d);
            matrix.divide(v_hat, w1d, w1);
            matrix.scalarMultiply(w1, alpha, aw1);
            matrix.subtract(w, aw1, w2);
            w = w2;
            util.print1DMatrix(w, features);
            //delete [] xi;
        }
    }

    cout << "============================================" << endl;
    printf("Final Weight\n");
    util.print1DMatrix(w,features);
    this->setWFinal(w);
    cout << "============================================" << endl;
}

double* SGD::getW() const {
    return w;
}

SGD::SGD(double beta1, double beta2, double **X, double *y, double alpha, int iterations, int features,
         int trainingSamples) : beta1(beta1), beta2(beta2), X(X), y(y), alpha(alpha), iterations(iterations),
                                features(features), trainingSamples(trainingSamples) {}

double *SGD::getWFinal() const {
    return wFinal;
}

void SGD::setWFinal(double *wFinal) {
    SGD::wFinal = wFinal;
}

SGD::SGD(double beta1, double beta2, double alpha, int iterations, int features, int trainingSamples,
         int testingSamples) : beta1(beta1), beta2(beta2), alpha(alpha), iterations(iterations), features(features),
                               trainingSamples(trainingSamples), testingSamples(testingSamples) {}
