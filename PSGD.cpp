//
// Created by vibhatha on 11/11/18.
//

#include "PSGD.h"
#include <iostream>
#include "Initializer.h"
#include "Util.h"
#include "Matrix.h"
#include <math.h>
#include <mpi.h>

using namespace std;

PSGD::PSGD(double **Xn, double* yn, double alphan, int itrN) {
    X = Xn;
    y = yn;
    alpha = alphan;
    iterations = itrN;
}

PSGD::PSGD(double **Xn, double *yn, double alphan, int itrN, int features_, int trainingSamples_, int testingSamples_) {
    X = Xn;
    y = yn;
    alpha = alphan;
    iterations = itrN;
    features = features_;
    trainingSamples = trainingSamples_;
    testingSamples = testingSamples_;
}

void PSGD::sgd() {
    Initializer initializer;
    wInit = initializer.initialWeights(features);
    double* wglobal = initializer.zeroWeights(features);
    Util util;
    //cout << "Training Samples : " << trainingSamples << endl;
    //util.print1DMatrix(wInit, features);
    Matrix matrix(features);
    w = wInit;
    for (int i = 0; i < iterations; ++i) {
        if(i%10 == 0 and world_rank==0) {
            cout << "Iteration " << i << "/" << iterations << endl;
        }
        for (int j = 0; j < trainingSamples; ++j) {
            double* xi = X[j];
            double yi = y[j];
            double yixiw = matrix.dot(xi, w) * yi;
            alpha = 1.0 / (1.0 + i);
            //cout << i << ", " << yixiw << endl;
            if(yixiw<1) {
                double* xiyia = matrix.scalarMultiply(matrix.subtract(w,matrix.scalarMultiply(xi, yi)), alpha);
                w = matrix.subtract(w, xiyia);
            } else {
                double* wa = matrix.scalarMultiply(w,alpha);
                w = matrix.subtract(w,wa);
            }
            MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
            w = matrix.scalarMultiply(wglobal, 1.0 / double(world_size));
            //util.print1DMatrix(w,features);
        }
    }
    this->setWFinal(w);
    //util.print1DMatrix(w,features);
    //printf("Final Weight\n");
}

void PSGD::adamSGD() {
    int totalSamples = trainingSamples;
    Initializer initializer;
    wInit = initializer.zeroWeights(features);
    double* v = initializer.zeroWeights(features);
    double* r = initializer.zeroWeights(features);
    double* v1 = initializer.zeroWeights(features);
    double* v2 = initializer.zeroWeights(features);
    double* r1 = initializer.zeroWeights(features);
    double* r2 = initializer.zeroWeights(features);
    double* w1 = initializer.zeroWeights(features);
    double* w2 = initializer.zeroWeights(features);
    double* wglobal = initializer.zeroWeights(features);
    double* v_hat = initializer.zeroWeights(features);
    double* r_hat = initializer.zeroWeights(features);
    double* grad_mul = initializer.zeroWeights(features);
    double* gradient = initializer.zeroWeights(features);
    double epsilon = 0.00000001;
    Util util;
    cout << "Beta 1 :" << beta1 << ", Beta 2 :" << beta2 << endl;
    //util.print1DMatrix(wInit, features);
    //util.print1DMatrix(v, features);
    //util.print1DMatrix(r, features);


    Matrix matrix(features);
    w = wInit;
    for (int i = 1; i < iterations; ++i) {
        if(world_rank==0){
            if(i % 10 == 0) {
                //cout << "+++++++++++++++++++++++++++++++++" << endl;
                //util.print1DMatrix(w, features);
                //cout << "+++++++++++++++++++++++++++++++++" << endl;
                cout << "Iteration " << i << "/" << iterations << endl;
            }
        }

        for (int j = 0; j < trainingSamples; ++j) {

            double yixiw = matrix.dot(X[j], w) * y[j];
            //cout << i << ", " << yixiw << endl;
            alpha = 1.0 / (1.0 + double(i));
            double coefficient = 1.0 /(1.0 + double(i));
            if(yixiw<1) {
                gradient = matrix.scalarMultiply(matrix.subtract(matrix.scalarMultiply(w, coefficient),matrix.scalarMultiply(X[j], y[j])), alpha);

            } else {
                gradient = matrix.scalarMultiply(matrix.scalarMultiply(w, coefficient), alpha);
            }

            v1 = matrix.scalarMultiply(v, beta1);
            v2 = matrix.scalarMultiply(gradient, (1-beta1));
            v = matrix.add(v1, v2);
            v_hat = matrix.scalarMultiply(v, (1.0 /(1.0-(pow(beta1,(double)i)))));
            grad_mul = matrix.inner(gradient, gradient);
            r1 = matrix.scalarMultiply(r, beta2);
            r2 = matrix.scalarMultiply(grad_mul, (1-beta2));
            r = matrix.add(r1, r2);
            r_hat = matrix.scalarMultiply(r, (1.0/(1.0-(pow(beta2,(double)i)))));
            w1 = matrix.divide(v_hat, matrix.scalarAddition(matrix.sqrt(r_hat),epsilon));
            w2 = matrix.scalarMultiply(w1, alpha);
            w = matrix.subtract(w,w2);
            MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
            w = matrix.scalarMultiply(wglobal, 1.0 / (double)world_size);
//            if(world_rank==0) {
//                //util.print1DMatrix(w, features);
//            }

            //delete [] xi;
        }
    }
    if(world_rank==0) {
        cout << "============================================" << endl;
        printf("Final Weight\n");
        util.print1DMatrix(w,features);
        this->setWFinal(w);
        cout << "============================================" << endl;
    }
    delete [] v ;delete [] r ;delete [] v1 ;delete [] v2 ;delete [] r1 ;delete [] r2 ;delete [] w1 ;delete [] w2 ;delete [] wglobal ;delete [] v_hat ;delete [] r_hat ;delete [] grad_mul ;delete [] gradient;

}

double* PSGD::getW() const {
    return w;
}

PSGD::PSGD(double beta1, double beta2, double **X, double *y, double alpha, int iterations, int features,
         int trainingSamples) : beta1(beta1), beta2(beta2), X(X), y(y), alpha(alpha), iterations(iterations),
                                features(features), trainingSamples(trainingSamples) {}

double *PSGD::getWFinal() const {
    return wFinal;
}

void PSGD::setWFinal(double *wFinal) {
    PSGD::wFinal = wFinal;
}

PSGD::PSGD(double beta1, double beta2, double alpha, int iterations, int features, int trainingSamples,
         int testingSamples) : beta1(beta1), beta2(beta2), alpha(alpha), iterations(iterations), features(features),
                               trainingSamples(trainingSamples), testingSamples(testingSamples) {}

PSGD::PSGD(double beta1, double beta2, double **X, double *y, double alpha, int iterations, int features,
           int world_size, int world_rank) : beta1(beta1), beta2(beta2), X(X), y(y), alpha(alpha),
                                             iterations(iterations), features(features), world_size(world_size),
                                             world_rank(world_rank) {}

PSGD::PSGD(double beta1, double beta2, double **X, double *y, double alpha, int iterations, int features,
           int trainingSamples, int testingSamples, int world_size, int world_rank) : beta1(beta1), beta2(beta2), X(X),
                                                                                      y(y), alpha(alpha),
                                                                                      iterations(iterations),
                                                                                      features(features),
                                                                                      trainingSamples(trainingSamples),
                                                                                      testingSamples(testingSamples),
                                                                                      world_size(world_size),
                                                                                      world_rank(world_rank) {}
