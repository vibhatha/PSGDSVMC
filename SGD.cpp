//
// Created by vibhatha on 11/5/18.
//

#include <iostream>
#include "SGD.h"
#include "Initializer.h"
#include "Util.h"
#include "Matrix.h"
#include "Matrix1.h"
#include "Predict.h"
#include <math.h>
#include <algorithm>

using namespace std;

SGD::SGD(double **Xn, double *yn, double alphan, int itrN) {
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
//    Initializer initializer;
//    wInit = initializer.initialWeights(features);
//    Util util;
//    cout << "Training Samples : " << trainingSamples << endl;
//    util.print1DMatrix(wInit, features);
//    Matrix matrix(features);
//    w = wInit;
//    double *res = new double[features];
//    for (int i = 0; i < iterations; ++i) {
//        if (i % 10 == 0) {
//            cout << "Iteration " << i << "/" << iterations << endl;
//        }
//        for (int j = 0; j < trainingSamples; ++j) {
//            double yixiw = matrix.dot(X[j], w) * y[j];
//            //cout << i << ", " << yixiw << endl;
//            if (yixiw < 1) {
//                double *xiyia = matrix.scalarMultiply(matrix.subtract(w, matrix.scalarMultiply(X[j], y[j], res), res),
//                                                      alpha, res);
//                w = matrix.subtract(w, xiyia, res);
//            } else {
//                double *wa = matrix.scalarMultiply(w, alpha, res);
//                w = matrix.subtract(w, wa, res);
//            }
//            //util.print1DMatrix(w,features);
//        }
//    }
//    this->setWFinal(w);
    //util.print1DMatrix(w,features);

    //printf("Final Weight\n");
}

void SGD::adamSGD() {
    Initializer initializer;
    double* w = new double[features];
    initializer.initializeWeightsWithArray(features, w);
    double *v = new double[features];
    initializer.initializeWeightsWithArray(features, wInit);
    initializer.initializeWeightsWithArray(features, v);
    double *r = new double[features];
    initializer.initializeWeightsWithArray(features, r);
    double *v1 = new double[features];
    initializer.initializeWeightsWithArray(features, v1);
    double *v2 = new double[features];
    initializer.initializeWeightsWithArray(features, v2);
    double *r1 = new double[features];
    initializer.initializeWeightsWithArray(features, v2);
    double *r2 = new double[features];
    initializer.initializeWeightsWithArray(features, v2);
    double *w1 = new double[features];
    initializer.initializeWeightsWithArray(features, w1);
    double *w2 = new double[features];
    initializer.initializeWeightsWithArray(features, w2);
    double *v_hat = new double[features];
    initializer.initializeWeightsWithArray(features, v_hat);
    double *r_hat = new double[features];
    initializer.initializeWeightsWithArray(features, r_hat);
    double *sq_r_hat = new double[features];
    initializer.initializeWeightsWithArray(features, sq_r_hat);
    double *grad_mul = new double[features];
    initializer.initializeWeightsWithArray(features, grad_mul);
    double *gradient = new double[features];
    initializer.initializeWeightsWithArray(features, gradient);
    double *xiyi = new double[features];
    initializer.initializeWeightsWithArray(features, xiyi);
    double *w_xiyi = new double[features];
    initializer.initializeWeightsWithArray(features, w_xiyi);
    double *aw_axiyi = new double[features];
    initializer.initializeWeightsWithArray(features, aw_axiyi);
    double *w1d = new double[features];
    initializer.initializeWeightsWithArray(features, w1d);
    double *aw1 = new double[features];
    initializer.initializeWeightsWithArray(features, aw1);
    double epsilon = 0.00000001;
    Util util;
    cout << "Training Samples : " << trainingSamples << endl;
    cout << "Beta 1 :" << beta1 << ", Beta 2 :" << beta2 << endl;
    //util.print1DMatrix(wInit, features);
    //util.print1DMatrix(v, features);
    //util.print1DMatrix(r, features);

    Matrix1 matrix(features);


    for (int i = 1; i < iterations; ++i) {
        if (i % 10 == 0) {
            //cout << "+++++++++++++++++++++++++++++++++" << endl;
            //util.print1DMatrix(w, features);
            //cout << "+++++++++++++++++++++++++++++++++" << endl;
            cout << "Iteration " << i << "/" << iterations << endl;
        }
        for (int j = 0; j < trainingSamples; ++j) {

            double yixiw = matrix.dot(X[j], w);
            yixiw = yixiw * y[j];

            double coefficient = 1.0 / (1.0 + double(i));

            if (yixiw < 1) {
                matrix.scalarAddition(X[j], y[j], xiyi);
                matrix.subtract(w, xiyi, w_xiyi);
                matrix.scalarMultiply(w_xiyi, alpha, gradient);

            } else {
                matrix.scalarMultiply(w, (1 - alpha), gradient);
            }


            matrix.scalarMultiply(v, beta1, v1);
            matrix.scalarMultiply(gradient, (1 - beta1), v2);
            matrix.add(v1, v2, v);
            matrix.scalarMultiply(v, (1.0 / (1.0 - (pow(beta1, (double) i)))), v_hat);
            matrix.inner(gradient, gradient, grad_mul);
            matrix.scalarMultiply(r, beta2, r1);
            matrix.scalarMultiply(grad_mul, (1 - beta2), r2);
            matrix.add(r1, r2, r);
            matrix.scalarMultiply(r, (1.0 / (1.0 - (pow(beta2, (double) i)))), r_hat);
            //w1 = matrix.sqrt(r_hat);
            //w1 = matrix.scalarAddition(w1,epsilon);
            //w1 = matrix.divide(v_hat,w1);
            matrix.sqrt(r_hat, sq_r_hat);
            matrix.scalarAddition(sq_r_hat, epsilon, w1d);
            matrix.divide(v_hat, w1d, w1);
            matrix.scalarMultiply(w1, alpha, aw1);
            matrix.subtract(w, aw1, w2);
            w = w2;
            //util.print1DMatrix(w, features);
            //delete [] xi;
        }
    }

    //cout << "============================================" << endl;
    //printf("Final Weight\n");
    //util.print1DMatrix(w, features);
    this->setWFinal(w);
    //cout << "============================================" << endl;
    delete [] v, v1, v2, r, r1, r2, v_hat, r_hat, w1, w2, grad_mul, sq_r_hat, gradient, w_xiyi, aw_axiyi, aw1, xiyi, w1d, wInit;
}

void SGD::adamSGD(double* w) {
    Initializer initializer;

    double *v = new double[features];
    initializer.initializeWeightsWithArray(features, v);
    double *r = new double[features];
    initializer.initializeWeightsWithArray(features, r);
    double *v1 = new double[features];
    initializer.initializeWeightsWithArray(features, v1);
    double *v2 = new double[features];
    initializer.initializeWeightsWithArray(features, v2);
    double *r1 = new double[features];
    initializer.initializeWeightsWithArray(features, v2);
    double *r2 = new double[features];
    initializer.initializeWeightsWithArray(features, v2);
    double *w1 = new double[features];
    initializer.initializeWeightsWithArray(features, w1);
    double *w2 = new double[features];
    initializer.initializeWeightsWithArray(features, w2);
    double *v_hat = new double[features];
    initializer.initializeWeightsWithArray(features, v_hat);
    double *r_hat = new double[features];
    initializer.initializeWeightsWithArray(features, r_hat);
    double *sq_r_hat = new double[features];
    initializer.initializeWeightsWithArray(features, sq_r_hat);
    double *grad_mul = new double[features];
    initializer.initializeWeightsWithArray(features, grad_mul);
    double *gradient = new double[features];
    initializer.initializeWeightsWithArray(features, gradient);
    double *xiyi = new double[features];
    initializer.initializeWeightsWithArray(features, xiyi);
    double *w_xiyi = new double[features];
    initializer.initializeWeightsWithArray(features, w_xiyi);
    double *aw_axiyi = new double[features];
    initializer.initializeWeightsWithArray(features, aw_axiyi);
    double *w1d = new double[features];
    initializer.initializeWeightsWithArray(features, w1d);
    double *aw1 = new double[features];
    initializer.initializeWeightsWithArray(features, aw1);
    double epsilon = 0.00000001;
    Util util;
    cout << "Training Samples : " << trainingSamples << endl;
    cout << "Beta 1 :" << beta1 << ", Beta 2 :" << beta2 << endl;
    //util.print1DMatrix(wInit, features);
    //util.print1DMatrix(v, features);
    //util.print1DMatrix(r, features);

    Matrix1 matrix(features);

    initializer.initializeWeightsWithArray(features, w);

    for (int i = 1; i < iterations; ++i) {
//        if (i % 10 == 0) {
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            //util.print1DMatrix(w, features);
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            cout << "Iteration " << i << "/" << iterations << endl;
//        }
        for (int j = 0; j < trainingSamples; ++j) {

            double yixiw = matrix.dot(X[j], w);
            yixiw = yixiw * y[j];

            double coefficient = 1.0 / (1.0 + double(i));

            if (yixiw < 1) {
                matrix.scalarAddition(X[j], y[j], xiyi);
                matrix.subtract(w, xiyi, w_xiyi);
                matrix.scalarMultiply(w_xiyi, alpha, gradient);

            } else {
                matrix.scalarMultiply(w, (1 - alpha), gradient);
            }


            matrix.scalarMultiply(v, beta1, v1);
            matrix.scalarMultiply(gradient, (1 - beta1), v2);
            matrix.add(v1, v2, v);
            matrix.scalarMultiply(v, (1.0 / (1.0 - (pow(beta1, (double) i)))), v_hat);
            matrix.inner(gradient, gradient, grad_mul);
            matrix.scalarMultiply(r, beta2, r1);
            matrix.scalarMultiply(grad_mul, (1 - beta2), r2);
            matrix.add(r1, r2, r);
            matrix.scalarMultiply(r, (1.0 / (1.0 - (pow(beta2, (double) i)))), r_hat);
            //w1 = matrix.sqrt(r_hat);
            //w1 = matrix.scalarAddition(w1,epsilon);
            //w1 = matrix.divide(v_hat,w1);
            matrix.sqrt(r_hat, sq_r_hat);
            matrix.scalarAddition(sq_r_hat, epsilon, w1d);
            matrix.divide(v_hat, w1d, w1);
            matrix.scalarMultiply(w1, alpha, aw1);
            matrix.subtract(w, aw1, w);
            //util.print1DMatrix(w, features);
            //delete [] xi;
            //check co-validation accuracy
        }
        Predict predict(Xtest, ytest, w , testingSamples, features);
        double acc = predict.predict();
        cout << "Epoch " << i << "Testing Accuracy : " << acc << "%" << endl;
    }

    cout << "============================================" << endl;
    printf("Final Weight\n");
    util.print1DMatrix(w, features);

    cout << "============================================" << endl;
    delete [] v;
    delete [] v1;
    delete [] v2;
    delete [] r;
    delete [] r1;
    delete [] r2;
    delete [] v_hat;
    delete [] r_hat;
    delete [] w1;
    delete [] w2;
    delete [] grad_mul;
    delete [] sq_r_hat;
    delete [] gradient;
    delete [] w_xiyi;
    delete [] aw_axiyi;
    delete [] aw1;
    delete [] xiyi;
    delete [] w1d;
}

void SGD::adamSGD(double *w, string summarylogfile, string epochlogfile) {
    Initializer initializer;

    double *v = new double[features];
    initializer.initializeWeightsWithArray(features, v);
    double *r = new double[features];
    initializer.initializeWeightsWithArray(features, r);
    double *v1 = new double[features];
    initializer.initializeWeightsWithArray(features, v1);
    double *v2 = new double[features];
    initializer.initializeWeightsWithArray(features, v2);
    double *r1 = new double[features];
    initializer.initializeWeightsWithArray(features, v2);
    double *r2 = new double[features];
    initializer.initializeWeightsWithArray(features, v2);
    double *w1 = new double[features];
    initializer.initializeWeightsWithArray(features, w1);
    double *w2 = new double[features];
    initializer.initializeWeightsWithArray(features, w2);
    double *v_hat = new double[features];
    initializer.initializeWeightsWithArray(features, v_hat);
    double *r_hat = new double[features];
    initializer.initializeWeightsWithArray(features, r_hat);
    double *sq_r_hat = new double[features];
    initializer.initializeWeightsWithArray(features, sq_r_hat);
    double *grad_mul = new double[features];
    initializer.initializeWeightsWithArray(features, grad_mul);
    double *gradient = new double[features];
    initializer.initializeWeightsWithArray(features, gradient);
    double *xiyi = new double[features];
    initializer.initializeWeightsWithArray(features, xiyi);
    double *w_xiyi = new double[features];
    initializer.initializeWeightsWithArray(features, w_xiyi);
    double *aw_axiyi = new double[features];
    initializer.initializeWeightsWithArray(features, aw_axiyi);
    double *w1d = new double[features];
    initializer.initializeWeightsWithArray(features, w1d);
    double *aw1 = new double[features];
    initializer.initializeWeightsWithArray(features, aw1);
    double epsilon = 0.00000001;
    Util util;
    cout << "Training Samples : " << trainingSamples << endl;
    cout << "Beta 1 :" << beta1 << ", Beta 2 :" << beta2 << endl;
    //util.print1DMatrix(wInit, features);
    //util.print1DMatrix(v, features);
    //util.print1DMatrix(r, features);

    Matrix1 matrix(features);

    initializer.initializeWeightsWithArray(features, w);

    for (int i = 1; i < iterations; ++i) {
//        if (i % 10 == 0) {
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            //util.print1DMatrix(w, features);
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            cout << "Iteration " << i << "/" << iterations << endl;
//        }
        //alpha = 1.0 / ((double)(i) + 1);
        //double coefficient = 1.0/(1.0 + (double)i);
        for (int j = 0; j < trainingSamples; ++j) {

            double yixiw = matrix.dot(X[j], w);
            yixiw = yixiw * y[j];



            if (yixiw < 1) {
                matrix.scalarMultiply(X[j], y[j], xiyi);
                matrix.subtract(w, xiyi, w_xiyi);
                matrix.scalarMultiply(w_xiyi, alpha, gradient);


            } else {
                matrix.scalarMultiply(w, (1 - alpha), gradient);
            }


            matrix.scalarMultiply(v, beta1, v1);
            matrix.scalarMultiply(gradient, (1 - beta1), v2);
            matrix.add(v1, v2, v);
            matrix.scalarMultiply(v, (1.0 / (1.0 - (pow(beta1, (double) i)))), v_hat);
            matrix.inner(gradient, gradient, grad_mul);
            matrix.scalarMultiply(r, beta2, r1);
            matrix.scalarMultiply(grad_mul, (1 - beta2), r2);
            matrix.add(r1, r2, r);
            matrix.scalarMultiply(r, (1.0 / (1.0 - (pow(beta2, (double) i)))), r_hat);
            //w1 = matrix.sqrt(r_hat);
            //w1 = matrix.scalarAddition(w1,epsilon);
            //w1 = matrix.divide(v_hat,w1);
            matrix.sqrt(r_hat, sq_r_hat);
            matrix.scalarAddition(sq_r_hat, epsilon, w1d);
            matrix.divide(v_hat, w1d, w1);
            matrix.scalarMultiply(w1, alpha, aw1);
            matrix.subtract(w, aw1, w);
            //util.print1DMatrix(w, features);
            //delete [] xi;
            //check co-validation accuracy
        }
        //util.print1DMatrix(w, features);
        Predict predict(Xtest, ytest, w , testingSamples, features);
        double acc = predict.predict();
        cout << "Epoch " << i << " Testing Accuracy : " << acc << "%" << endl;
        util.writeAccuracyPerEpoch(i, acc, epochlogfile);
    }

//    cout << "============================================" << endl;
//    printf("Final Weight\n");
//    util.print1DMatrix(w, features);
//
//    cout << "============================================" << endl;
    delete [] v;
    delete [] v1;
    delete [] v2;
    delete [] r;
    delete [] r1;
    delete [] r2;
    delete [] v_hat;
    delete [] r_hat;
    delete [] w1;
    delete [] w2;
    delete [] grad_mul;
    delete [] sq_r_hat;
    delete [] gradient;
    delete [] w_xiyi;
    delete [] aw_axiyi;
    delete [] aw1;
    delete [] xiyi;
    delete [] w1d;
}

void SGD::sgd(double *w, string summarylogfile, string epochlogfile) {

    Initializer initializer;
    double *v = new double[features];
    initializer.initializeWeightsWithArray(features, v);
    double *r = new double[features];
    initializer.initializeWeightsWithArray(features, r);
    double *v1 = new double[features];
    initializer.initializeWeightsWithArray(features, v1);
    double *v2 = new double[features];
    initializer.initializeWeightsWithArray(features, v2);
    double *r1 = new double[features];
    initializer.initializeWeightsWithArray(features, v2);
    double *r2 = new double[features];
    initializer.initializeWeightsWithArray(features, v2);
    double *w1 = new double[features];
    initializer.initializeWeightsWithArray(features, w1);
    double *w2 = new double[features];
    initializer.initializeWeightsWithArray(features, w2);
    double *v_hat = new double[features];
    initializer.initializeWeightsWithArray(features, v_hat);
    double *r_hat = new double[features];
    initializer.initializeWeightsWithArray(features, r_hat);
    double *sq_r_hat = new double[features];
    initializer.initializeWeightsWithArray(features, sq_r_hat);
    double *grad_mul = new double[features];
    initializer.initializeWeightsWithArray(features, grad_mul);
    double *gradient = new double[features];
    initializer.initializeWeightsWithArray(features, gradient);
    double *xiyi = new double[features];
    initializer.initializeWeightsWithArray(features, xiyi);
    double *w_xiyi = new double[features];
    initializer.initializeWeightsWithArray(features, w_xiyi);
    double *aw_axiyi = new double[features];
    initializer.initializeWeightsWithArray(features, aw_axiyi);
    double *w1d = new double[features];
    initializer.initializeWeightsWithArray(features, w1d);
    double *aw1 = new double[features];
    initializer.initializeWeightsWithArray(features, aw1);
    double epsilon = 0.00000001;
    Util util;

    Matrix1 matrix(features);

    initializer.initialWeights(features, w);

    for (int i = 1; i < iterations; ++i) {
//        if (i % 10 == 0) {
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            //util.print1DMatrix(w, features);
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            cout << "Iteration " << i << "/" << iterations << endl;
//        }
        //alpha = 1.0 / ((double)(i) + 1);
        //double coefficient = 1.0/(1.0 + (double)i);
        for (int j = 0; j < trainingSamples; ++j) {

            double yixiw = matrix.dot(X[j], w);
            yixiw = yixiw * y[j];

            if (yixiw < 1) {
                matrix.scalarMultiply(X[j], y[j], xiyi);
                matrix.subtract(w, xiyi, w_xiyi);
                matrix.scalarMultiply(w_xiyi, alpha, gradient);
            } else {
                matrix.scalarMultiply(w, (1 - alpha), gradient);
            }

            matrix.scalarMultiply(gradient, alpha, aw1);
            matrix.subtract(w, aw1, w);
            //util.print1DMatrix(w, 5);
        }
        Predict predict(Xtest, ytest, w , testingSamples, features);
        double acc = predict.predict();
        cout << "SGD Epoch " << i << " Testing Accuracy : " << acc << "%" << endl;
        util.writeAccuracyPerEpoch(i, acc, epochlogfile);
    }

    delete [] gradient;
    delete [] w_xiyi;
    delete [] aw1;
    delete [] xiyi;

}

void SGD::pegasosSgd(double *w, string summarylogfile, string epochlogfile) {

    Initializer initializer;
    double *w1 = new double[features];
    initializer.initializeWeightsWithArray(features, w1);
    double *xiyi = new double[features];
    initializer.initializeWeightsWithArray(features, xiyi);
    double epsilon = 0.00000001;
    double eta = 0;
    clock_t prediction_time;

    double totalpredictiontime=0;

    Util util;

    Matrix1 matrix(features);

    initializer.initialWeights(features, w);
    double cost = 1.0;
    int i=1;
    while (true) {
        eta = 1.0 / (alpha * i);
//        if (i % 10 == 0) {
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            //util.print1DMatrix(w, features);
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            cout << "Iteration " << i << "/" << iterations << endl;
//        }
        //alpha = 1.0 / ((double)(i) + 1);
        //double coefficient = 1.0/(1.0 + (double)i);
        for (int j = 0; j < trainingSamples; ++j) {

            double yixiw = matrix.dot(X[j], w);
            yixiw = yixiw * y[j];

            if (yixiw < 1) {
                matrix.scalarMultiply(X[j], y[j]*eta, xiyi);
                matrix.scalarMultiply(w, (1-(eta*alpha)), w1);
                matrix.add(w1, xiyi, w);
            } else {
                matrix.scalarMultiply(w, (1 - (eta*alpha)), w);
            }
            cost = 0.5 * alpha * fabs(matrix.dot(w,w)) + max(0.0, (1-yixiw));
            //util.print1DMatrix(w, 5);
        }
        prediction_time = clock();
        Predict predict(Xtest, ytest, w , testingSamples, features);
        double acc = predict.predict();
        cout << "Pegasos SGD Epoch " << i << " Testing Accuracy : " << acc << "%" << ", Hinge Loss : " << cost << endl;
        util.writeAccuracyPerEpoch(i, acc, epochlogfile);
        prediction_time = clock()-prediction_time;
        totalpredictiontime += (((double)prediction_time)/CLOCKS_PER_SEC);
        i++;
        if(cost<0.1){
            break;
        }
    }

    this->setTotalPredictionTime(totalpredictiontime);

    delete [] xiyi;
    delete [] w1;
}


double *SGD::getW() const {
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

SGD::SGD(double beta1, double beta2, double **X, double *y, double *w, double alpha, int iterations, int features,
         int trainingSamples, int testingSamples, double **Xtest, double *ytest) : beta1(beta1), beta2(beta2), X(X),
                                                                                   y(y), w(w), alpha(alpha),
                                                                                   iterations(iterations),
                                                                                   features(features),
                                                                                   trainingSamples(trainingSamples),
                                                                                   testingSamples(testingSamples),
                                                                                   Xtest(Xtest), ytest(ytest) {}

double SGD::getTotalPredictionTime() const {
    return totalPredictionTime;
}

void SGD::setTotalPredictionTime(double totalPredictionTime) {
    SGD::totalPredictionTime = totalPredictionTime;
}
