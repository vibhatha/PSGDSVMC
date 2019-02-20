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
#include <vector>
#include <chrono>
#include <random>

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
    double *w = new double[features];
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
    delete[] v, v1, v2, r, r1, r2, v_hat, r_hat, w1, w2, grad_mul, sq_r_hat, gradient, w_xiyi, aw_axiyi, aw1, xiyi, w1d, wInit;
}

void SGD::adamSGD(double *w) {
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
        Predict predict(Xtest, ytest, w, testingSamples, features);
        double acc = predict.predict();
        cout << "Epoch " << i << "Testing Accuracy : " << acc << "%" << endl;
    }

    cout << "============================================" << endl;
    printf("Final Weight\n");
    util.print1DMatrix(w, features);

    cout << "============================================" << endl;
    delete[] v;
    delete[] v1;
    delete[] v2;
    delete[] r;
    delete[] r1;
    delete[] r2;
    delete[] v_hat;
    delete[] r_hat;
    delete[] w1;
    delete[] w2;
    delete[] grad_mul;
    delete[] sq_r_hat;
    delete[] gradient;
    delete[] w_xiyi;
    delete[] aw_axiyi;
    delete[] aw1;
    delete[] xiyi;
    delete[] w1d;
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
        Predict predict(Xtest, ytest, w, testingSamples, features);
        double acc = predict.predict();
        cout << "Epoch " << i << " Testing Accuracy : " << acc << "%" << endl;
        util.writeAccuracyPerEpoch(i, acc, epochlogfile);
    }

//    cout << "============================================" << endl;
//    printf("Final Weight\n");
//    util.print1DMatrix(w, features);
//
//    cout << "============================================" << endl;
    delete[] v;
    delete[] v1;
    delete[] v2;
    delete[] r;
    delete[] r1;
    delete[] r2;
    delete[] v_hat;
    delete[] r_hat;
    delete[] w1;
    delete[] w2;
    delete[] grad_mul;
    delete[] sq_r_hat;
    delete[] gradient;
    delete[] w_xiyi;
    delete[] aw_axiyi;
    delete[] aw1;
    delete[] xiyi;
    delete[] w1d;
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
        Predict predict(Xtest, ytest, w, testingSamples, features);
        double acc = predict.predict();
        cout << "SGD Epoch " << i << " Testing Accuracy : " << acc << "%" << endl;
        util.writeAccuracyPerEpoch(i, acc, epochlogfile);
    }

    delete[] gradient;
    delete[] w_xiyi;
    delete[] aw1;
    delete[] xiyi;

}

void SGD::pegasosSgd(double *w, string summarylogfile, string epochlogfile) {

    double init_time = 0;
    init_time -= clock();
    Initializer initializer;
    double *w1 = new double[features];
    initializer.initializeWeightsWithArray(features, w1);
    double *xiyi = new double[features];
    initializer.initializeWeightsWithArray(features, xiyi);
    double epsilon = 0.00000001;
    double eta = 0;
    clock_t prediction_time;

    double totalpredictiontime = 0;

    Util util;

    Matrix1 matrix(features);

    //generate a random seed of data points
    vector<int> accuracies_set(0);
    vector<int> indices(trainingSamples);
    std::iota(indices.begin(), indices.end(), 0);
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();

    shuffle(indices.begin(), indices.end(), default_random_engine(seed));


    initializer.initialWeights(features, w);
    double cost = 1.0;
    double error_threshold = this->getError_threshold();
    int i = 1;
    double error = 100;
    int marker = 0;
    double cost_sum = 0;
    init_time += clock();
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
        int j = 0;
        double dot_prod_time = 0;
        double weight_update_time = 0;
        double cost_calculate_time = 0;
        double convergence_calculate_time = 0;
        double predict_time = 0;
        double log_write_time = 0;
        for (int k = 0; k < trainingSamples; ++k) {
            j = indices.at(k);

            dot_prod_time -= clock();
            double yixiw = matrix.dot(X[j], w);
            yixiw = yixiw * y[j];
            dot_prod_time += clock();
            weight_update_time -= clock();
            if (yixiw < 1) {
                matrix.scalarMultiply(X[j], y[j] * eta, xiyi);
                matrix.scalarMultiply(w, (1 - (eta * alpha)), w1);
                matrix.add(w1, xiyi, w);
            } else {
                matrix.scalarMultiply(w, (1 - (eta * alpha)), w);
            }
            weight_update_time += clock();
            cost_calculate_time -= clock();
            cost = 0.5 * alpha * fabs(matrix.dot(w, w)) + max(0.0, (1 - yixiw));
            cost_sum += cost;
            cost_calculate_time += clock();
            //util.print1DMatrix(w, 5);
        }
        prediction_time = clock();
        Predict predict(Xtest, ytest, w, testingSamples, features);
        cost = cost_sum / trainingSamples;
        cost_sum = 0;
        double acc = predict.predict();
        prediction_time = clock() - prediction_time;
        convergence_calculate_time -= clock();
        i++;
        error = 100.0 - acc;
        if (cost < error_threshold) {
            accuracies_set.push_back(marker);
        } else {
            marker = 0;
            accuracies_set.clear();
        }

        if (accuracies_set.size() == 5 or i > iterations) {
            break;
        }
        convergence_calculate_time += clock();
        totalpredictiontime += (((double) prediction_time) / CLOCKS_PER_SEC);
        dot_prod_time /= CLOCKS_PER_SEC;
        weight_update_time /= CLOCKS_PER_SEC;
        cost_calculate_time /= CLOCKS_PER_SEC;
        convergence_calculate_time /= CLOCKS_PER_SEC;
        prediction_time /= CLOCKS_PER_SEC;
        cout << "Pegasos SGD Epoch " << i << " Testing Accuracy : " << acc << "%" << ", Hinge Loss : " << cost << endl;
        log_write_time -= clock();
        util.writeAccuracyPerEpoch(i, acc, dot_prod_time, weight_update_time, cost_calculate_time,
                                   convergence_calculate_time, prediction_time, epochlogfile);
        log_write_time += clock();
        totalpredictiontime += (((double) log_write_time) / CLOCKS_PER_SEC);
    }
    totalpredictiontime += (((double) init_time) / CLOCKS_PER_SEC);
    this->setTotalPredictionTime(totalpredictiontime);


    delete[] xiyi;
    delete[] w1;
}


void SGD::pegasosSgdNoTiming(double *w, string summarylogfile, string epochlogfile) {
    double init_time = 0;
    init_time -= clock();
    Initializer initializer;
    double *w1 = new double[features];
    initializer.initializeWeightsWithArray(features, w1);
    double *xiyi = new double[features];
    initializer.initializeWeightsWithArray(features, xiyi);
    double epsilon = 0.00000001;
    double eta = 0;
    clock_t prediction_time;

    double totalpredictiontime = 0;

    Util util;

    Matrix1 matrix(features);

    //generate a random seed of data points
    vector<int> accuracies_set(0);
    vector<int> indices(trainingSamples);
    std::iota(indices.begin(), indices.end(), 0);
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();

    shuffle(indices.begin(), indices.end(), default_random_engine(seed));


    initializer.initialWeights(features, w);
    double cost = 1.0;
    double error_threshold = this->getError_threshold();
    int i = 1;
    double error = 100;
    int marker = 0;
    double cost_sum = 0;
    init_time += clock();
    double yixiw = 0;
    int j = 0;
    while (true) {
        eta = 1.0 / (alpha * i);

        for (int k = 0; k < trainingSamples; ++k) {
            j = indices.at(k);
            yixiw = matrix.dot(X[j], w);
            yixiw = yixiw * y[j];
            if (yixiw < 1) {
                matrix.scalarMultiply(X[j], y[j] * eta, xiyi);
                matrix.scalarMultiply(w, (1 - (eta * alpha)), w1);
                matrix.add(w1, xiyi, w);
            } else {
                matrix.scalarMultiply(w, (1 - (eta * alpha)), w);
            }
            //util.print1DMatrix(w, 5);
        }
        prediction_time = clock();
        Predict predict(Xtest, ytest, w, testingSamples, features);
        cost = (0.5 * alpha * fabs(matrix.dot(w, w))) + max(0.0, (1 - yixiw));
        //cost = cost_sum / trainingSamples;
        double acc = predict.crossValidate();
        util.writeAccuracyPerEpoch(i, acc, epochlogfile);
        prediction_time = clock() - prediction_time;
        i++;
        error = 100.0 - acc;
        if (cost < error_threshold) {
            accuracies_set.push_back(marker);
        } else {
            marker = 0;
            accuracies_set.clear();
        }

        if (accuracies_set.size() == 5 or i > iterations) {
            break;
        }

        totalpredictiontime += (((double) prediction_time) / CLOCKS_PER_SEC);
        cout << "Pegasos SGD Epoch " << i << " Testing Accuracy : " << acc << "%" << ", Hinge Loss : " << cost << endl;
    }
    totalpredictiontime += (((double) init_time) / CLOCKS_PER_SEC);
    this->setTotalPredictionTime(totalpredictiontime);


    delete[] xiyi;
    delete[] w1;
}


void SGD::pegasosBlockSgd(double *w, string summarylogfile, string epochlogfile, int block_size) {
    cout << "Pegasos Block SGD " << endl;
    Initializer initializer;
    double *w1 = new double[features];
    initializer.initializeWeightsWithArray(features, w1);
    double *w_init = new double[features];
    initializer.initialWeights(features, w_init);
    double *xiyi = new double[features];
    initializer.initializeWeightsWithArray(features, xiyi);
    double *tempW = new double[features];
    initializer.initializeWeightsWithArray(features, tempW);
    double epsilon = 0.00000001;
    double eta = 0;
    clock_t prediction_time;

    double totalpredictiontime = 0;

    Util util;

    Matrix1 matrix(features);

    //generate a random seed of data points
    vector<int> accuracies_set(0);
    vector<int> indices(trainingSamples);
    std::iota(indices.begin(), indices.end(), 0);
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();

    shuffle (indices.begin(), indices.end(), default_random_engine(seed));

    initializer.initialWeights(features, w);
    w_init = w;
    double cost = 1.0;
    double acc = 0;
    double error_threshold = this->getError_threshold();
    int i = 1;
    double error = 100;
    int marker = 0;
    double cost_sum = 0;
    while (true) {
        eta = 1.0 / (alpha * i);
        int j = 0;
        double yixiw = 0;
        int count = 0;
        for (int k = 0; k < trainingSamples; k = k + block_size) {
            //cout << "---------------------" << endl;
            for (int l = 0; l < block_size; l++) {
                j = indices.at(k + l);
                yixiw = matrix.dot(X[j], w);
                yixiw = yixiw * y[j];
                //cout << i << ", " << j << " : " << X[j][0] << " , " << y[j] << ", " << yixiw << endl;
                if (yixiw < 1) {
                    matrix.scalarMultiply(X[j], y[j] * eta, xiyi);
                    matrix.scalarMultiply(w, (1 - (eta * alpha)), w1);
                    matrix.add(w1, xiyi, w);
                } else {
                    matrix.scalarMultiply(w, (1 - (eta * alpha)), w);
                }
                // util.print1DMatrix(w, features);
                for (int m = 0; m < features; ++m) {
                    tempW[m] += w[m];
                }
                util.copyArray(w_init, w, features);
                //w=w_init;
                count++;
                if (count > trainingSamples -
                            1) { // idea is when there is the last block that size can be lesser than the block_size, so we have to stop when the final element is processed and move to next iteration
                    break;
                }
            }
            //cout << "---------------------" << endl;
            for (int m = 0; m < features; ++m) {
                w[m] = tempW[m] / (double) block_size;
            }

            util.copyArray(w, w_init, features);
            //w_init = w;

            //util.print1DMatrix(w, 5);
            //util.print1DMatrix(w, features);
            initializer.initializeWeightsWithArray(features, tempW);
        }
        prediction_time = clock();
        Predict predict(Xtest, ytest, w, testingSamples, features);
        //cost = cost_sum / trainingSamples;
        //cost_sum = 0;
        cost = (0.5 * alpha * fabs(matrix.dot(w, w))) + max(0.0, (1 - yixiw));
        acc = predict.crossValidate();
        cout << "Block Size:  " << block_size << ", Pegasos Block SGD Epoch " << i << " Testing Accuracy : " << acc
             << "%" << ", Hinge Loss : " << cost << ", Count : " << count << "/" << trainingSamples << endl;

        util.writeAccuracyPerEpoch(i, acc, epochlogfile);
        i++;
        error = 100.0 - acc;
        prediction_time = clock() - prediction_time;
        if (cost < error_threshold) {
            accuracies_set.push_back(marker);
        } else {
            marker = 0;
            accuracies_set.clear();
        }

        if (accuracies_set.size() == 5 or iterations < i) {
            break;
        }

        totalpredictiontime += (((double) prediction_time) / CLOCKS_PER_SEC);
    }

    this->setTotalPredictionTime(totalpredictiontime);
    this->setEffective_epochs(i);
    this->setResultant_minimum_cost(cost);
    this->setResultant_cross_accuracy(acc);

    delete[] xiyi;
    delete[] w1;
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

double SGD::getError_threshold() const {
    return error_threshold;
}

void SGD::setError_threshold(double error_threshold) {
    SGD::error_threshold = error_threshold;
}

int SGD::getEffective_epochs() const {
    return effective_epochs;
}

void SGD::setEffective_epochs(int effective_epochs) {
    SGD::effective_epochs = effective_epochs;
}

double SGD::getResultant_minimum_cost() const {
    return resultant_minimum_cost;
}

void SGD::setResultant_minimum_cost(double resultant_minimum_cost) {
    SGD::resultant_minimum_cost = resultant_minimum_cost;
}

double SGD::getResultant_cross_accuracy() const {
    return resultant_cross_accuracy;
}

void SGD::setResultant_cross_accuracy(double resultant_cross_accuracy) {
    SGD::resultant_cross_accuracy = resultant_cross_accuracy;
}

int seed() {
    int i = 1;
    return i++;
}