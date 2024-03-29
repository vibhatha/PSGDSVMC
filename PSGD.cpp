//
// Created by vibhatha on 11/11/18.
//

#include "PSGD.h"
#include <iostream>
#include <fstream>
#include "Initializer.h"
#include "Util.h"
#include "Matrix.h"
#include "Matrix1.h"
#include "Predict.h"
#include <math.h>
#include <mpi.h>
#include <random>
#include <chrono>
#include <algorithm>
#include "omp.h"
#include "cblas.h"


using namespace std;
using namespace arma;



PSGD::PSGD(double **Xn, double *yn, double alphan, int itrN) {
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
    double *wglobal = initializer.zeroWeights(features);
    Util util;
    //cout << "Training Samples : " << trainingSamples << endl;
    //util.print1DMatrix(wInit, features);
    Matrix matrix(features);
    double *res = new double[features];
    w = wInit;
    for (int i = 0; i < iterations; ++i) {
        if (i % 10 == 0 and world_rank == 0) {
            cout << "Iteration " << i << "/" << iterations << endl;
        }
        for (int j = 0; j < trainingSamples; ++j) {
            double *xi = X[j];
            double yi = y[j];
            double yixiw = matrix.dot(xi, w) * yi;
            alpha = 1.0 / (1.0 + i);
            //cout << i << ", " << yixiw << endl;
            if (yixiw < 1) {
                double *xiyia = matrix.scalarMultiply(matrix.subtract(w, matrix.scalarMultiply(xi, yi, res), res),
                                                      alpha, res);
                w = matrix.subtract(w, xiyia, res);
            } else {
                double *wa = matrix.scalarMultiply(w, alpha, res);
                w = matrix.subtract(w, wa, res);
            }

            MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
            w = matrix.scalarMultiply(wglobal, 1.0 / double(world_size), res);
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
    double *v = initializer.zeroWeights(features);
    double *r = initializer.zeroWeights(features);
    double *v1 = initializer.zeroWeights(features);
    double *v2 = initializer.zeroWeights(features);
    double *r1 = initializer.zeroWeights(features);
    double *r2 = initializer.zeroWeights(features);
    double *w1 = initializer.zeroWeights(features);
    double *w2 = initializer.zeroWeights(features);
    double *wglobal = initializer.zeroWeights(features);
    double *v_hat = initializer.zeroWeights(features);
    double *r_hat = initializer.zeroWeights(features);
    double *grad_mul = initializer.zeroWeights(features);
    double *gradient = initializer.zeroWeights(features);
    double epsilon = 0.1;
    Util util;
    //cout << "Beta 1 :" << beta1 << ", Beta 2 :" << beta2 << endl;
    //util.print1DMatrix(wInit, features);
    //util.print1DMatrix(v, features);
    //util.print1DMatrix(r, features);

    double *res = new double[features];
    Matrix matrix(features);
    w = wInit;
    compute_time = 0;
    communication_time = 0;

    for (int i = 1; i < iterations; ++i) {
        if (world_rank == 0) {
            if (i % 10 == 0) {
                //cout << "+++++++++++++++++++++++++++++++++" << endl;
                //util.print1DMatrix(w, features);
                //cout << "+++++++++++++++++++++++++++++++++" << endl;
                cout << "Iteration " << i << "/" << iterations << endl;
            }
        }

        for (int j = 0; j < trainingSamples; ++j) {
            double start_compute = MPI_Wtime();
            double yixiw = matrix.dot(X[j], w) * y[j];
            //cout << i << ", " << yixiw << endl;
            double coefficient = 1.0 / (1.0 + double(i));

            if (yixiw < 1) {
                gradient = matrix.scalarMultiply(matrix.subtract(matrix.scalarMultiply(w, coefficient, res),
                                                                 matrix.scalarMultiply(X[j], y[j], res), res), alpha,
                                                 res);

            } else {
                gradient = matrix.scalarMultiply(matrix.scalarMultiply(w, coefficient, res), alpha, res);
            }

            v1 = matrix.scalarMultiply(v, beta1, res);
            v2 = matrix.scalarMultiply(gradient, (1 - beta1), res);
            v = matrix.add(v1, v2, res);
            v_hat = matrix.scalarMultiply(v, (1.0 / (1.0 - (pow(beta1, (double) i)))), res);
            grad_mul = matrix.inner(gradient, gradient, res);
            r1 = matrix.scalarMultiply(r, beta2, res);
            r2 = matrix.scalarMultiply(grad_mul, (1 - beta2), res);
            r = matrix.add(r1, r2, res);
            r_hat = matrix.scalarMultiply(r, (1.0 / (1.0 - (pow(beta2, (double) i)))), res);
            w1 = matrix.divide(v_hat, matrix.scalarAddition(matrix.sqrt(r_hat, res), epsilon, res), res);
            w2 = matrix.scalarMultiply(w1, alpha, res);
            w = matrix.subtract(w, w2, res);
            double end_compute = MPI_Wtime();
            compute_time += (end_compute - compute_time);
            double start_communication = MPI_Wtime();
            MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
            double end_communication = MPI_Wtime();
            communication_time += (end_communication - start_communication);
            start_compute = MPI_Wtime();
            w = matrix.scalarMultiply(wglobal, 1.0 / (double) world_size, res);
            end_compute = MPI_Wtime();
            compute_time += (end_compute - compute_time);
//            if(world_rank==0) {
//                //util.print1DMatrix(w, features);
//            }

            //delete [] xi;
            //delete [] v ;delete [] r ;delete [] v1 ;delete [] v2 ;delete [] r1 ;delete [] r2 ;delete [] w1 ;delete [] w2 ;delete [] wglobal ;delete [] v_hat ;delete [] r_hat ;delete [] grad_mul ;delete [] gradient;
        }
    }
    if (world_rank == 0) {
        cout << "============================================" << endl;
        printf("Final Weight\n");
        util.print1DMatrix(w, features);
        //this->setWFinal(w);
        cout << "============================================" << endl;
    }


}

void PSGD::adamSGDSeq(double *w) {
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
    //cout << "Training Samples : " << trainingSamples << endl;
    //cout << "Beta 1 :" << beta1 << ", Beta 2 :" << beta2 << endl;
    //util.print1DMatrix(wInit, features);
    //util.print1DMatrix(v, features);
    //util.print1DMatrix(r, features);

    Matrix1 matrix(features);

    initializer.initializeWeightsWithArray(features, w);
    compute_time = 0;
    communication_time = 0;
    double start_compute = 0;
    for (int i = 1; i < iterations; ++i) {
        /*if (i % 10 == 0 and world_rank==0) {
            //cout << "+++++++++++++++++++++++++++++++++" << endl;
            //util.print1DMatrix(w, features);
            //cout << "+++++++++++++++++++++++++++++++++" << endl;
            cout << "Iteration " << i << "/" << iterations << endl;
        }*/
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
        }
    }
    /*if(world_rank==0) {
        cout << "============================================" << endl;
        printf("Final Weight\n");
        util.print1DMatrix(w, features);

        cout << "============================================" << endl;
    }*/

    //cout << "Compute Time of Rank : " << world_rank << " is " << compute_time << endl;
    //cout << "Communication Time of Rank : " << world_rank << " is " << communication_time << endl;

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

void PSGD::adamSGD(double *w) {
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
    double *wglobal = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal);
    double epsilon = 0.00000001;
    Util util;
    //cout << "Training Samples : " << trainingSamples << endl;
    //cout << "Beta 1 :" << beta1 << ", Beta 2 :" << beta2 << endl;
    //util.print1DMatrix(wInit, features);
    //util.print1DMatrix(v, features);
    //util.print1DMatrix(r, features);

    Matrix1 matrix(features);

    initializer.initializeWeightsWithArray(features, w);
    compute_time = 0;
    communication_time = 0;
    double start_compute = 0;
    for (int i = 1; i < iterations; ++i) {
        /*if (i % 10 == 0 and world_rank==0) {
            //cout << "+++++++++++++++++++++++++++++++++" << endl;
            //util.print1DMatrix(w, features);
            //cout << "+++++++++++++++++++++++++++++++++" << endl;
            cout << "Iteration " << i << "/" << iterations << endl;
        }*/
        for (int j = 0; j < trainingSamples; ++j) {
            start_compute = MPI_Wtime();
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
            double end_compute = MPI_Wtime();
            compute_time += (end_compute - end_compute);
            double start_communication = MPI_Wtime();
            MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
            double end_communication = MPI_Wtime();
            communication_time += (end_communication - start_communication);
            start_compute = MPI_Wtime();
            matrix.scalarMultiply(wglobal, 1.0 / (double) world_size, w);
            end_compute = MPI_Wtime();
            compute_time += (end_compute - start_compute);
            //util.print1DMatrix(w, features);
            //delete [] xi;
        }
    }
    /*if(world_rank==0) {
        cout << "============================================" << endl;
        printf("Final Weight\n");
        util.print1DMatrix(w, features);

        cout << "============================================" << endl;
    }*/

    //cout << "Compute Time of Rank : " << world_rank << " is " << compute_time << endl;
    //cout << "Communication Time of Rank : " << world_rank << " is " << communication_time << endl;

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

void PSGD::adamSGD(double *w, string logfile) {
    Initializer initializer;
    cout << "Start Training ..." << endl;
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
    double *wglobal = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal);
    double epsilon = 0.00000001;
    Util util;
    //cout << "Training Samples : " << trainingSamples << endl;
    //cout << "Beta 1 :" << beta1 << ", Beta 2 :" << beta2 << endl;
    //util.print1DMatrix(wInit, features);
    //util.print1DMatrix(v, features);
    //util.print1DMatrix(r, features);

    Matrix1 matrix(features);

    initializer.initializeWeightsWithArray(features, w);
    compute_time = 0;
    communication_time = 0;
    double start_compute = 0;


    double **comptimeA;
    double **commtimeA;
    comptimeA = new double *[iterations];
    for (int i = 0; i < iterations; ++i) {
        comptimeA[i] = new double[trainingSamples];
    }

    commtimeA = new double *[iterations];
    for (int i = 0; i < iterations; ++i) {
        commtimeA[i] = new double[trainingSamples];
    }

    for (int i = 1; i < iterations; ++i) {
        if (i % 10 == 0 and world_rank == 0) {
            //cout << "+++++++++++++++++++++++++++++++++" << endl;
            //util.print1DMatrix(w, features);
            //cout << "+++++++++++++++++++++++++++++++++" << endl;
            cout << "Iteration " << i << "/" << iterations << endl;
        }
        commtimeA[i] = new double[trainingSamples];
        comptimeA[i] = new double[trainingSamples];
        for (int j = 0; j < trainingSamples; ++j) {

            double perDataPerItrCompt = 0;
            start_compute = MPI_Wtime();
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
            double end_compute = MPI_Wtime();
            compute_time += (end_compute - start_compute);
            perDataPerItrCompt = (end_compute - start_compute);
            double start_communication = MPI_Wtime();
            MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
            double end_communication = MPI_Wtime();
            communication_time += (end_communication - start_communication);
            start_compute = MPI_Wtime();
            matrix.scalarMultiply(wglobal, 1.0 / (double) world_size, w);
            end_compute = MPI_Wtime();
            compute_time += (end_compute - start_compute);
            perDataPerItrCompt += (end_compute - start_compute);
            //util.print1DMatrix(w, features);
            //delete [] xi;
            commtimeA[i][j] = end_communication - start_communication;
            comptimeA[i][j] = perDataPerItrCompt;

        }
    }
    /*if(world_rank==0) {
        cout << "============================================" << endl;
        printf("Final Weight\n");
        util.print1DMatrix(w, features);

        cout << "============================================" << endl;
    }*/
    cout << "Compute Time of Rank : " << world_rank << " is " << compute_time << endl;
    cout << "Communication Time of Rank : " << world_rank << " is " << communication_time << endl;
    writeLog(logfile.append("_process=").append(to_string(world_rank)), iterations, trainingSamples, comptimeA,
             commtimeA);


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
    delete[] commtimeA;
    delete[] comptimeA;
}

void PSGD::adamSGDSeq(double *w, string logfile) {
    Initializer initializer;
    cout << "Start Training ..." << endl;
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

    Matrix1 matrix(features);

    initializer.initializeWeightsWithArray(features, w);
    for (int i = 1; i < iterations; ++i) {
        if (i % 10 == 0 and world_rank == 0) {
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
            matrix.sqrt(r_hat, sq_r_hat);
            matrix.scalarAddition(sq_r_hat, epsilon, w1d);
            matrix.divide(v_hat, w1d, w1);
            matrix.scalarMultiply(w1, alpha, aw1);
            matrix.subtract(w, aw1, w);
        }
    }

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

void PSGD::adamSGDBatchv1(double *w) {
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
    double *wglobal = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal);
    double epsilon = 0.00000001;
    Util util;
    //cout << "Training Samples : " << trainingSamples << endl;
    //cout << "Beta 1 :" << beta1 << ", Beta 2 :" << beta2 << endl;
    //util.print1DMatrix(wInit, features);
    //util.print1DMatrix(v, features);
    //util.print1DMatrix(r, features);

    Matrix1 matrix(features);

    initializer.initializeWeightsWithArray(features, w);
    compute_time = 0;
    communication_time = 0;
    double start_compute = 0;
    for (int i = 1; i < iterations; ++i) {

//        if (i % 10 == 0 and world_rank==0) {
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            //util.print1DMatrix(w, features);
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            cout << "Iteration " << i << "/" << iterations << endl;
//        }
        for (int j = 0; j < trainingSamples; ++j) {
            start_compute = MPI_Wtime();
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
            matrix.sqrt(r_hat, sq_r_hat);
            matrix.scalarAddition(sq_r_hat, epsilon, w1d);
            matrix.divide(v_hat, w1d, w1);
            matrix.scalarMultiply(w1, alpha, aw1);
            matrix.subtract(w, aw1, w);
            double end_compute = MPI_Wtime();
            compute_time += (end_compute - start_compute);
            //util.print1DMatrix(w, features);
            //delete [] xi;
        }
        double start_communication = MPI_Wtime();
        MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        matrix.scalarMultiply(wglobal, 1.0 / (double) world_size, w);
        double end_communication = MPI_Wtime();
        communication_time += (end_communication - start_communication);
    }
    /*if(world_rank==0) {
        cout << "============================================" << endl;
        printf("Final Weight\n");
        util.print1DMatrix(w, features);

        cout << "============================================" << endl;
    }*/

    //cout << "Compute Time of Rank : " << world_rank << " is " << compute_time << endl;
    //cout << "Communication Time of Rank : " << world_rank << " is " << communication_time << endl;

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


void PSGD::adamSGDBatchv1(double *w, string logfile) {
    Initializer initializer;
    cout << "Start Training ..." << endl;
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
    double *wglobal = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal);
    double epsilon = 0.00000001;
    Util util;
    //cout << "Training Samples : " << trainingSamples << endl;
    //cout << "Beta 1 :" << beta1 << ", Beta 2 :" << beta2 << endl;
    //util.print1DMatrix(wInit, features);
    //util.print1DMatrix(v, features);
    //util.print1DMatrix(r, features);

    Matrix1 matrix(features);

    initializer.initializeWeightsWithArray(features, w);
    compute_time = 0;
    communication_time = 0;
    double start_compute = 0;


    double **comptimeA;
    double *commtimeA;
    comptimeA = new double *[iterations];
    for (int i = 0; i < iterations; ++i) {
        comptimeA[i] = new double[trainingSamples];
    }

    commtimeA = new double[iterations];


    for (int i = 1; i < iterations; ++i) {
//        if (i % 10 == 0 and world_rank==0) {
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            //util.print1DMatrix(w, features);
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            cout << "Iteration " << i << "/" << iterations << endl;
//        }
        comptimeA[i] = new double[trainingSamples];
        for (int j = 0; j < trainingSamples; ++j) {

            double perDataPerItrCompt = 0;
            start_compute = MPI_Wtime();
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
            double end_compute = MPI_Wtime();
            compute_time += (end_compute - start_compute);
            perDataPerItrCompt = (end_compute - start_compute);
            comptimeA[i][j] = perDataPerItrCompt;
        }
        double start_communication = MPI_Wtime();
        MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        double end_communication = MPI_Wtime();
        matrix.scalarMultiply(wglobal, 1.0 / (double) world_size, w);
        communication_time += (end_communication - start_communication);
        commtimeA[i] = end_communication - start_communication;
    }
    /*if(world_rank==0) {
        cout << "============================================" << endl;
        printf("Final Weight\n");
        util.print1DMatrix(w, features);

        cout << "============================================" << endl;
    }*/
    cout << "Compute Time of Rank : " << world_rank << " is " << compute_time << endl;
    cout << "Communication Time of Rank : " << world_rank << " is " << communication_time << endl;
    writeLog(logfile.append("_process=").append(to_string(world_rank)), iterations, trainingSamples, comptimeA,
             commtimeA);


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
    delete[] commtimeA;
    delete[] comptimeA;
}


void PSGD::adamSGDBatchv2(double *w, int comm_gap) {
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
    double *wglobal = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal);
    double epsilon = 0.00000001;
    Util util;
    //cout << "Training Samples : " << trainingSamples << endl;
    //cout << "Beta 1 :" << beta1 << ", Beta 2 :" << beta2 << endl;
    //util.print1DMatrix(wInit, features);
    //util.print1DMatrix(v, features);
    //util.print1DMatrix(r, features);
    if (world_rank == 0) {
        cout << "Gap : " << comm_gap << ", Data Size : " << trainingSamples << endl;
    }
    Matrix1 matrix(features);

    initializer.initializeWeightsWithArray(features, w);
    compute_time = 0;
    communication_time = 0;
    double start_compute = 0;
    for (int i = 1; i < iterations; ++i) {
        //alpha = 1.0 / (1.0 + (double) i);
        /*if (i % 10 == 0 and world_rank==0) {
            //cout << "+++++++++++++++++++++++++++++++++" << endl;
            //util.print1DMatrix(w, features);
            //cout << "+++++++++++++++++++++++++++++++++" << endl;
            cout << "Iteration " << i << "/" << iterations << endl;
        }*/
        for (int j = 0; j < trainingSamples; ++j) {
            start_compute = MPI_Wtime();
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
            double end_compute = MPI_Wtime();
            compute_time += (end_compute - end_compute);
            if (j % comm_gap == 0) {
                double start_communication = MPI_Wtime();
                MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
                double end_communication = MPI_Wtime();
                communication_time += (end_communication - start_communication);
                start_compute = MPI_Wtime();
                matrix.scalarMultiply(wglobal, 1.0 / (double) world_size, w);
                end_compute = MPI_Wtime();
                compute_time += (end_compute - start_compute);
            }

            //util.print1DMatrix(w, features);
            //delete [] xi;
        }
    }
    /*if(world_rank==0) {
        cout << "============================================" << endl;
        printf("Final Weight\n");
        util.print1DMatrix(w, features);

        cout << "============================================" << endl;
    }*/

    //cout << "Compute Time of Rank : " << world_rank << " is " << compute_time << endl;
    //cout << "Communication Time of Rank : " << world_rank << " is " << communication_time << endl;

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


void PSGD::adamSGDBatchv2(double *w, int comm_gap, string logfile) {
    Initializer initializer;
    cout << "Start Training ..." << endl;
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
    double *wglobal = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal);
    double epsilon = 0.00000001;
    Util util;
    //cout << "Training Samples : " << trainingSamples << endl;
    //cout << "Beta 1 :" << beta1 << ", Beta 2 :" << beta2 << endl;
    //util.print1DMatrix(wInit, features);
    //util.print1DMatrix(v, features);
    //util.print1DMatrix(r, features);

    Matrix1 matrix(features);

    initializer.initializeWeightsWithArray(features, w);
    compute_time = 0;
    communication_time = 0;
    double start_compute = 0;


    double **comptimeA;
    double **commtimeA;
    comptimeA = new double *[iterations];
    for (int i = 0; i < iterations; ++i) {
        comptimeA[i] = new double[trainingSamples];
    }

    commtimeA = new double *[iterations];
    for (int i = 0; i < iterations; ++i) {
        commtimeA[i] = new double[trainingSamples];
    }

    for (int i = 1; i < iterations; ++i) {
        alpha = 1.0 / (1.0 + (double) i);
        if (i % 10 == 0 and world_rank == 0) {
            //cout << "+++++++++++++++++++++++++++++++++" << endl;
            //util.print1DMatrix(w, features);
            //cout << "+++++++++++++++++++++++++++++++++" << endl;
            cout << "Iteration " << i << "/" << iterations << endl;
        }
        commtimeA[i] = new double[trainingSamples];
        comptimeA[i] = new double[trainingSamples];
        for (int j = 0; j < trainingSamples; ++j) {

            double perDataPerItrCompt = 0;
            start_compute = MPI_Wtime();
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
            double end_compute = MPI_Wtime();
            compute_time += (end_compute - start_compute);
            perDataPerItrCompt = (end_compute - start_compute);
            if (j % comm_gap == 0) {
                double start_communication = MPI_Wtime();
                MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
                double end_communication = MPI_Wtime();
                communication_time += (end_communication - start_communication);
                start_compute = MPI_Wtime();
                matrix.scalarMultiply(wglobal, 1.0 / (double) world_size, w);
                end_compute = MPI_Wtime();
                compute_time += (end_compute - start_compute);
                perDataPerItrCompt += (end_compute - start_compute);
                commtimeA[i][j] = end_communication - start_communication;
            }
            //util.print1DMatrix(w, features);
            //delete [] xi;
            commtimeA[i][j] = 0;
            comptimeA[i][j] = perDataPerItrCompt;

        }
    }
    /*if(world_rank==0) {
        cout << "============================================" << endl;
        printf("Final Weight\n");
        util.print1DMatrix(w, features);

        cout << "============================================" << endl;
    }*/
    cout << "Compute Time of Rank : " << world_rank << " is " << compute_time << endl;
    cout << "Communication Time of Rank : " << world_rank << " is " << communication_time << endl;
    writeLog(logfile.append("_process=").append(to_string(world_rank)), iterations, trainingSamples, comptimeA,
             commtimeA);


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
    delete[] commtimeA;
    delete[] comptimeA;
}

void PSGD::adamSGDBatchv2(double *w, int comm_gap, string logfile, string epochlogfile) {
    Initializer initializer;
    cout << "Start Training ..." << endl;
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
    double *w_print = new double[features];
    initializer.initializeWeightsWithArray(features, w_print);
    double *aw_axiyi = new double[features];
    initializer.initializeWeightsWithArray(features, aw_axiyi);
    double *w1d = new double[features];
    initializer.initializeWeightsWithArray(features, w1d);
    double *aw1 = new double[features];
    initializer.initializeWeightsWithArray(features, aw1);
    double *wglobal = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal);
    double *wglobal_print = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal_print);
    double epsilon = 0.00000001;
    Util util;
    //cout << "Training Samples : " << trainingSamples << endl;
    //cout << "Beta 1 :" << beta1 << ", Beta 2 :" << beta2 << endl;
    //util.print1DMatrix(wInit, features);
    //util.print1DMatrix(v, features);
    //util.print1DMatrix(r, features);

    Matrix1 matrix(features);

    initializer.initializeWeightsWithArray(features, w);
    compute_time = 0;
    communication_time = 0;
    double start_compute = 0;


    double **comptimeA;
    double **commtimeA;
    comptimeA = new double *[iterations];
    for (int i = 0; i < iterations; ++i) {
        comptimeA[i] = new double[trainingSamples];
    }

    commtimeA = new double *[iterations];
    for (int i = 0; i < iterations; ++i) {
        commtimeA[i] = new double[trainingSamples];
    }

    for (int i = 1; i < iterations; ++i) {
        //alpha = 1.0 / (1.0 + (double) i);
//        if (i % 10 == 0 and world_rank == 0) {
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            //util.print1DMatrix(w, features);
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            cout << "Iteration " << i << "/" << iterations << endl;
//        }
        commtimeA[i] = new double[trainingSamples];
        comptimeA[i] = new double[trainingSamples];
        for (int j = 0; j < trainingSamples; ++j) {

            double perDataPerItrCompt = 0;
            start_compute = MPI_Wtime();
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
            double end_compute = MPI_Wtime();
            compute_time += (end_compute - start_compute);
            perDataPerItrCompt = (end_compute - start_compute);
            if (j % comm_gap == 0) {
                double start_communication = MPI_Wtime();
                MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
                double end_communication = MPI_Wtime();
                communication_time += (end_communication - start_communication);
                start_compute = MPI_Wtime();
                matrix.scalarMultiply(wglobal, 1.0 / (double) world_size, w);
                end_compute = MPI_Wtime();
                compute_time += (end_compute - start_compute);
                perDataPerItrCompt += (end_compute - start_compute);
                commtimeA[i][j] = end_communication - start_communication;
            }
            //util.print1DMatrix(w, features);
            //delete [] xi;
            commtimeA[i][j] = 0;
            comptimeA[i][j] = perDataPerItrCompt;

        }
        MPI_Allreduce(w, wglobal_print, features, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        matrix.scalarMultiply(wglobal_print, 1.0 / (double) world_size, w_print);
        if (world_rank == 0) {
            Predict predict(Xtest, ytest, w_print, testingSamples, features);
            double acc = predict.predict();
            cout << "PSGD AllReduce Adam Epoch " << i << " Testing Accuracy : " << acc << "%" << endl;
            util.writeAccuracyPerEpoch(i, acc, epochlogfile);
            util.print1DMatrix(w_print, 10);
        }

    }
    /*if(world_rank==0) {
        cout << "============================================" << endl;
        printf("Final Weight\n");
        util.print1DMatrix(w, features);

        cout << "============================================" << endl;
    }*/
    cout << "Compute Time of Rank : " << world_rank << " is " << compute_time << endl;
    cout << "Communication Time of Rank : " << world_rank << " is " << communication_time << endl;
    writeLog(logfile.append("_process=").append(to_string(world_rank)), iterations, trainingSamples, comptimeA,
             commtimeA);


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
    delete[] commtimeA;
    delete[] comptimeA;
}


void PSGD::sgdBatchv2(double *w, int comm_gap, string logfile, string epochlogfile) {
    Initializer initializer;
    cout << "Start Training ..." << endl;

    double *gradient = new double[features];
    initializer.initializeWeightsWithArray(features, gradient);
    double *xiyi = new double[features];
    initializer.initializeWeightsWithArray(features, xiyi);
    double *w_xiyi = new double[features];
    initializer.initializeWeightsWithArray(features, w_xiyi);
    double *aw_axiyi = new double[features];
    initializer.initializeWeightsWithArray(features, aw_axiyi);
    double *aw1 = new double[features];
    initializer.initializeWeightsWithArray(features, aw1);
    double *wglobal = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal);
    double epsilon = 0.00000001;
    Util util;
    //cout << "Training Samples : " << trainingSamples << endl;
    //cout << "Beta 1 :" << beta1 << ", Beta 2 :" << beta2 << endl;
    //util.print1DMatrix(wInit, features);
    //util.print1DMatrix(v, features);
    //util.print1DMatrix(r, features);

    Matrix1 matrix(features);

    initializer.initializeWeightsWithArray(features, w);
    compute_time = 0;
    communication_time = 0;
    double start_compute = 0;


    double **comptimeA;
    double **commtimeA;
    comptimeA = new double *[iterations];
    for (int i = 0; i < iterations; ++i) {
        comptimeA[i] = new double[trainingSamples];
    }

    commtimeA = new double *[iterations];
    for (int i = 0; i < iterations; ++i) {
        commtimeA[i] = new double[trainingSamples];
    }

    for (int i = 1; i < iterations; ++i) {
        //alpha = 1.0 / (1.0 + (double) i);
//        if (i % 10 == 0 and world_rank == 0) {
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            //util.print1DMatrix(w, features);
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            cout << "Iteration " << i << "/" << iterations << endl;
//        }
        commtimeA[i] = new double[trainingSamples];
        comptimeA[i] = new double[trainingSamples];
        for (int j = 0; j < trainingSamples; ++j) {

            double perDataPerItrCompt = 0;
            start_compute = MPI_Wtime();
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
            double end_compute = MPI_Wtime();
            compute_time += (end_compute - start_compute);
            perDataPerItrCompt = (end_compute - start_compute);
            if (j % comm_gap == 0) {
                double start_communication = MPI_Wtime();
                MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
                double end_communication = MPI_Wtime();
                communication_time += (end_communication - start_communication);
                start_compute = MPI_Wtime();
                matrix.scalarMultiply(wglobal, 1.0 / (double) world_size, w);
                end_compute = MPI_Wtime();
                compute_time += (end_compute - start_compute);
                perDataPerItrCompt += (end_compute - start_compute);
                commtimeA[i][j] = end_communication - start_communication;
            }
            //util.print1DMatrix(w, features);
            //delete [] xi;
            commtimeA[i][j] = 0;
            comptimeA[i][j] = perDataPerItrCompt;

        }

        util.print1DMatrix(w, 5);
        Predict predict(Xtest, ytest, w, testingSamples, features);
        double acc = predict.predict();
        cout << "PSGD Batch v2 Epoch " << i << " Testing Accuracy : " << acc << "%" << endl;
        util.writeAccuracyPerEpoch(i, acc, epochlogfile);


    }
    /*if(world_rank==0) {
        cout << "============================================" << endl;
        printf("Final Weight\n");
        util.print1DMatrix(w, features);

        cout << "============================================" << endl;
    }*/
    cout << "Compute Time of Rank : " << world_rank << " is " << compute_time << endl;
    cout << "Communication Time of Rank : " << world_rank << " is " << communication_time << endl;
    //writeLog(logfile.append("_process=").append(to_string(world_rank)), iterations, trainingSamples, comptimeA,
    //         commtimeA);


    delete[] gradient;
    delete[] w_xiyi;
    delete[] aw1;
    delete[] xiyi;
    delete[] commtimeA;
    delete[] comptimeA;
}

void PSGD::adamSGDFullBatchv1(double *w, string logfile, string epochlogfile) {
    Initializer initializer;
    cout << "Start Training ..." << endl;
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
    double *wglobal = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal);
    double epsilon = 0.00000001;
    Util util;
    //cout << "Training Samples : " << trainingSamples << endl;
    //cout << "Beta 1 :" << beta1 << ", Beta 2 :" << beta2 << endl;
    //util.print1DMatrix(wInit, features);
    //util.print1DMatrix(v, features);
    //util.print1DMatrix(r, features);

    Matrix1 matrix(features);

    initializer.initializeWeightsWithArray(features, w);
    compute_time = 0;
    communication_time = 0;
    double start_compute = 0;
    double final_communication_time = 0;

    double **comptimeA;

    comptimeA = new double *[iterations];
    for (int i = 0; i < iterations; ++i) {
        comptimeA[i] = new double[trainingSamples];
    }

    double predictionTime = 0;
    double perDataPerItrCompt = 0;
    for (int i = 1; i < iterations; ++i) {
        //alpha = 1.0 / (1.0 + (double) i);
//        if (i % 10 == 0 and world_rank == 0) {
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            //util.print1DMatrix(w, features);
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            cout << "Iteration " << i << "/" << iterations << endl;
//        }
        comptimeA[i] = new double[trainingSamples];
        for (int j = 0; j < trainingSamples; ++j) {

            perDataPerItrCompt = 0;
            start_compute = MPI_Wtime();
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
            double end_compute = MPI_Wtime();
            compute_time += (end_compute - start_compute);
            perDataPerItrCompt = (end_compute - start_compute);
            comptimeA[i][j] = perDataPerItrCompt;

        }
        double start_predict = MPI_Wtime();
        Predict predict(Xtest, ytest, w, testingSamples, features);
        double acc = predict.predict();
        double end_predict = MPI_Wtime();
        predictionTime += (end_predict - start_predict);
        cout << "PSGD Full Batch :  Epoch " << i << " Testing Accuracy : " << acc << "%" << endl;
        util.writeAccuracyPerEpoch(i, acc, epochlogfile);
    }

    double start_communication = MPI_Wtime();
    MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    double end_communication = MPI_Wtime();
    communication_time += (end_communication - start_communication);
    start_compute = MPI_Wtime();
    matrix.scalarMultiply(wglobal, 1.0 / (double) world_size, w);
    double end_compute = MPI_Wtime();
    compute_time += (end_compute - start_compute);

    cout << "Compute Time of Rank : " << world_rank << " is " << compute_time << endl;
    cout << "Communication Time of Rank : " << world_rank << " is " << communication_time << endl;
    writeLog(logfile.append("_process=").append(to_string(world_rank)), iterations, trainingSamples, comptimeA,
             communication_time);
    this->setTotalPredictionTime(predictionTime);

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
    delete[] comptimeA;
}


void PSGD::adamSGDFullBatchv2(double *w, string epochlogfile) {
    Initializer initializer;
    cout << "Start Training ..." << endl;
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
    double *wglobal = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal);
    double epsilon = 0.00000001;
    Util util;
    //cout << "Training Samples : " << trainingSamples << endl;
    //cout << "Beta 1 :" << beta1 << ", Beta 2 :" << beta2 << endl;
    //util.print1DMatrix(wInit, features);
    //util.print1DMatrix(v, features);
    //util.print1DMatrix(r, features);

    Matrix1 matrix(features);

    initializer.initializeWeightsWithArray(features, w);
    compute_time = 0;
    communication_time = 0;
    double start_compute = 0;
    double final_communication_time = 0;
    double predictionTime = 0;
    double perDataPerItrCompt = 0;
    for (int i = 1; i < iterations; ++i) {
        //alpha = 1.0 / (1.0 + (double) i);
//        if (i % 10 == 0 and world_rank == 0) {
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            //util.print1DMatrix(w, features);
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            cout << "Iteration " << i << "/" << iterations << endl;
//        }
        for (int j = 0; j < trainingSamples; ++j) {

            perDataPerItrCompt = 0;
            start_compute = MPI_Wtime();
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
            double end_compute = MPI_Wtime();
            compute_time += (end_compute - start_compute);
        }
        double start_predict = MPI_Wtime();
        Predict predict(Xtest, ytest, w, testingSamples, features);
        double acc = predict.predict();
        double end_predict = MPI_Wtime();
        predictionTime += (end_predict - start_predict);
        cout << "PSGD Full Batch : " << world_rank << " Epoch " << i << " Testing Accuracy : " << acc << "%" << endl;
        util.writeAccuracyPerEpoch(i, acc, epochlogfile);
    }

    double start_communication = MPI_Wtime();
    MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    double end_communication = MPI_Wtime();
    communication_time += (end_communication - start_communication);
    start_compute = MPI_Wtime();
    matrix.scalarMultiply(wglobal, 1.0 / (double) world_size, w);
    double end_compute = MPI_Wtime();
    compute_time += (end_compute - start_compute);

    cout << "Compute Time of Rank : " << world_rank << " is " << compute_time << endl;
    cout << "Communication Time of Rank : " << world_rank << " is " << communication_time << endl;
    this->setTotalPredictionTime(predictionTime);

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


void PSGD::sgdFullBatchv1(double *w, string logfile, string epochlogfile) {
    Initializer initializer;
    cout << "Start Training ..." << endl;

    double *gradient = new double[features];
    initializer.initializeWeightsWithArray(features, gradient);
    double *xiyi = new double[features];
    initializer.initializeWeightsWithArray(features, xiyi);
    double *aw1 = new double[features];
    initializer.initializeWeightsWithArray(features, aw1);
    double *wglobal = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal);
    double epsilon = 0.00000001;
    Util util;
    //cout << "Training Samples : " << trainingSamples << endl;
    //cout << "Beta 1 :" << beta1 << ", Beta 2 :" << beta2 << endl;
    //util.print1DMatrix(wInit, features);
    //util.print1DMatrix(v, features);
    //util.print1DMatrix(r, features);

    Matrix1 matrix(features);

    initializer.initializeWeightsWithArray(features, w);
    compute_time = 0;
    communication_time = 0;
    double start_compute = 0;
    double final_communication_time = 0;

    double **comptimeA;

    comptimeA = new double *[iterations];
    for (int i = 0; i < iterations; ++i) {
        comptimeA[i] = new double[trainingSamples];
    }

    double predictionTime = 0;
    double perDataPerItrCompt = 0;
    for (int i = 1; i < iterations; ++i) {
        //alpha = 1.0 / (1.0 + (double) i);
//        if (i % 10 == 0 and world_rank == 0) {
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            //util.print1DMatrix(w, features);
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            cout << "Iteration " << i << "/" << iterations << endl;
//        }
        comptimeA[i] = new double[trainingSamples];
        for (int j = 0; j < trainingSamples; ++j) {

            perDataPerItrCompt = 0;
            start_compute = MPI_Wtime();
            double yixiw = matrix.dot(X[j], w);
            yixiw = yixiw * y[j];

            if (yixiw < 1) {
                matrix.scalarAddition(X[j], y[j], xiyi);
                matrix.subtract(w, xiyi, gradient);

            } else {
                gradient = w;
            }

            matrix.scalarMultiply(gradient, alpha, aw1);
            matrix.subtract(w, aw1, w);
            double end_compute = MPI_Wtime();
            compute_time += (end_compute - start_compute);
            perDataPerItrCompt = (end_compute - start_compute);
            comptimeA[i][j] = perDataPerItrCompt;

        }
        double start_predict = MPI_Wtime();
        Predict predict(Xtest, ytest, w, testingSamples, features);
        double acc = predict.predict();
        double end_predict = MPI_Wtime();
        predictionTime += (end_predict - start_predict);
        cout << "PSGD Full Batch :  Epoch " << i << " Testing Accuracy : " << acc << "%" << endl;
        util.writeAccuracyPerEpoch(i, acc, epochlogfile);
    }

    double start_communication = MPI_Wtime();
    MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    double end_communication = MPI_Wtime();
    communication_time += (end_communication - start_communication);
    start_compute = MPI_Wtime();
    matrix.scalarMultiply(wglobal, 1.0 / (double) world_size, w);
    double end_compute = MPI_Wtime();
    compute_time += (end_compute - start_compute);

    cout << "Compute Time of Rank : " << world_rank << " is " << compute_time << endl;
    cout << "Communication Time of Rank : " << world_rank << " is " << communication_time << endl;
    writeLog(logfile.append("_process=").append(to_string(world_rank)), iterations, trainingSamples, comptimeA,
             communication_time);
    this->setTotalPredictionTime(predictionTime);


    delete[] gradient;
    delete[] aw1;
    delete[] xiyi;
    delete[] comptimeA;
    delete[] wglobal;
}


void PSGD::sgdFullBatchv2(double *w, string epochlogfile) {
    double C = 1.0;
    Initializer initializer;
    cout << "Start Training ..." << endl;
    double *gradient = new double[features];
    initializer.initializeWeightsWithArray(features, gradient);
    double *xiyi = new double[features];
    initializer.initializeWeightsWithArray(features, xiyi);
    double *aw1 = new double[features];
    initializer.initializeWeightsWithArray(features, aw1);
    double *wglobal = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal);
    double epsilon = 0.00000001;
    Util util;
    //cout << "Training Samples : " << trainingSamples << endl;
    //cout << "Beta 1 :" << beta1 << ", Beta 2 :" << beta2 << endl;
    //util.print1DMatrix(wInit, features);
    //util.print1DMatrix(v, features);
    //util.print1DMatrix(r, features);

    Matrix1 matrix(features);

    initializer.initializeWeightsWithArray(features, w);
    compute_time = 0;
    communication_time = 0;
    double start_compute = 0;
    double final_communication_time = 0;
    double predictionTime = 0;
    double perDataPerItrCompt = 0;
    for (int i = 1; i < iterations; ++i) {
        //alpha = 1.0 / (1.0 + (double) i);
//        if (i % 10 == 0 and world_rank == 0) {
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            //util.print1DMatrix(w, features);
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            cout << "Iteration " << i << "/" << iterations << endl;
//        }
        for (int j = 0; j < trainingSamples; ++j) {

            perDataPerItrCompt = 0;
            start_compute = MPI_Wtime();
            double yixiw = matrix.dot(X[j], w);
            yixiw = yixiw * y[j] * C;
            double coefficient = 1.0 / (1.0 + double(i));

            if (yixiw < 1) {
                matrix.scalarMultiply(X[j], C * y[j], xiyi);
                matrix.subtract(w, xiyi, gradient);

            } else {
                gradient = w;
            }

            matrix.scalarMultiply(gradient, alpha, aw1);
            matrix.subtract(w, aw1, w);
            double end_compute = MPI_Wtime();
            compute_time += (end_compute - start_compute);
        }
        double start_predict = MPI_Wtime();
        Predict predict(Xtest, ytest, w, testingSamples, features);
        double acc = predict.predict();
        double end_predict = MPI_Wtime();
        predictionTime += (end_predict - start_predict);
        cout << "PSGD Full Batch : " << world_rank << " Epoch " << i << " Testing Accuracy : " << acc << "%" << endl;
        util.writeAccuracyPerEpoch(i, acc, epochlogfile);
    }

    double start_communication = MPI_Wtime();
    MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    double end_communication = MPI_Wtime();
    communication_time += (end_communication - start_communication);
    start_compute = MPI_Wtime();
    matrix.scalarMultiply(wglobal, 1.0 / (double) world_size, w);
    double end_compute = MPI_Wtime();
    compute_time += (end_compute - start_compute);

    cout << "Compute Time of Rank : " << world_rank << " is " << compute_time << endl;
    cout << "Communication Time of Rank : " << world_rank << " is " << communication_time << endl;
    this->setTotalPredictionTime(predictionTime);


    delete[] gradient;
    delete[] aw1;
    delete[] xiyi;
    delete[] wglobal;
}

void PSGD::pegasosSGDFullBatchv1(double *w, string epochlogfile) {
    Initializer initializer;
    double *w1 = new double[features];
    initializer.initializeWeightsWithArray(features, w1);
    double *xiyi = new double[features];
    initializer.initializeWeightsWithArray(features, xiyi);
    double *wglobal = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal);
    double *local_cost = new double[1];
    initializer.initializeWeightsWithArray(1, local_cost);
    double *global_cost = new double[1];
    initializer.initializeWeightsWithArray(1, global_cost);
    double *wglobal_print = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal_print);
    double *w_print = new double[features];
    initializer.initializeWeightsWithArray(features, w_print);
    double epsilon = 0.00000001;
    double eta = 0;

    Util util;

    Matrix1 matrix(features);

    initializer.initialWeights(features, w);
    double error_threshold = this->getError_threshold();
    double error = 0;
    int breakFlag[] = {100};
    int i = 1;
    double predict_time = 0;
    double cost = 0;
    while (breakFlag[0] != -1) {
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
            double start_compute = MPI_Wtime();
            double yixiw = matrix.dot(X[j], w);
            yixiw = yixiw * y[j];

            if (yixiw < 1) {
                matrix.scalarMultiply(X[j], y[j] * eta, xiyi);
                matrix.scalarMultiply(w, (1 - (eta * alpha)), w1);
                matrix.add(w1, xiyi, w);
            } else {
                matrix.scalarMultiply(w, (1 - (eta * alpha)), w);
            }
            double end_compute = MPI_Wtime();
            compute_time += (end_compute - start_compute);
            //util.print1DMatrix(w, 5);
            cost = 0.5 * alpha * fabs(matrix.dot(w, w)) + max(0.0, (1 - yixiw));
            double start_cost = MPI_Wtime();
            local_cost[0] = cost;
            MPI_Allreduce(local_cost, global_cost, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            cost = global_cost[0] / (double) world_size;
            double end_cost = MPI_Wtime();
            predict_time += (end_cost - start_cost);
        }
//        double start_predict = MPI_Wtime();
//        Predict predict(Xtest, ytest, w , testingSamples, features);
//        double acc = predict.predict();
//        cout << "Pegasos Full Batch PSGD Epoch : Rank : " << world_rank << ", Epoch " << i << " Testing Accuracy : " << acc << "%" << endl;
//        util.writeAccuracyPerEpoch(i, acc, epochlogfile);
//        double end_predict = MPI_Wtime();
//        predict_time+= (end_predict-start_predict);
        double start_predict = MPI_Wtime();
        MPI_Allreduce(w, wglobal_print, features, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        matrix.scalarMultiply(wglobal_print, 1.0 / (double) world_size, w_print);
        if (world_rank == 0 and i % 100 == 0) {
            Predict predict(Xtest, ytest, w_print, testingSamples, features);
            double acc = predict.predict();
            error = 100.0 - acc;
            cout << "Pegasos Full Batch PSGD Epoch : Rank : " << world_rank << ", Epoch " << i << " Testing Accuracy : "
                 << acc << "%" << ", Hinge Loss : " << cost << endl;
            util.writeLossAccuracyPerEpoch(i, acc, cost, epochlogfile);
        }
        double end_predict = MPI_Wtime();
        predict_time += (end_predict - start_predict);
        if (error < error_threshold and world_rank == 0) {
            breakFlag[0] = -1;
        }
        double bcast_time_start = MPI_Wtime();
        MPI_Bcast(breakFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (breakFlag[0] == -1) {
            cout << "World Rank : " << world_rank << "Break Flag : " << breakFlag[0] << endl;
        }
        double bcast_time_end = MPI_Wtime();
        predict_time += (bcast_time_end - bcast_time_start);
        i++;
    }

    double start_communication = MPI_Wtime();
    MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    double end_communication = MPI_Wtime();
    communication_time += (end_communication - start_communication);
    double start_compute = MPI_Wtime();
    matrix.scalarMultiply(wglobal, 1.0 / (double) world_size, w);
    double end_compute = MPI_Wtime();
    compute_time += (end_compute - start_compute);

    this->setTotalPredictionTime(predict_time);

    delete[] xiyi;
    delete[] w1;
}

void PSGD::pegasosSGDFullBatchv2(double *w, string epochlogfile) {
    Initializer initializer;
    double *w1 = new double[features];
    initializer.initializeWeightsWithArray(features, w1);
    double *xiyi = new double[features];
    initializer.initializeWeightsWithArray(features, xiyi);
    double *wglobal = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal);
    double *local_cost = new double[1];
    initializer.initializeWeightsWithArray(1, local_cost);
    double *global_cost = new double[1];
    initializer.initializeWeightsWithArray(1, global_cost);
    double *wglobal_print = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal_print);
    double *w_print = new double[features];
    initializer.initializeWeightsWithArray(features, w_print);
    double epsilon = 0.00000001;
    double eta = 0;

    Util util;

    Matrix1 matrix(features);

    initializer.initialWeights(features, w);
    double error_threshold = this->getError_threshold();
    double error = 0;
    int breakFlag[] = {100};
    int i = 1;
    double predict_time = 0;
    double cost = 0;

    while (breakFlag[0] != -1) {
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
            double start_compute = MPI_Wtime();
            double yixiw = matrix.dot(X[j], w);
            yixiw = yixiw * y[j];

            if (yixiw < 1) {
                matrix.scalarMultiply(X[j], y[j] * eta, xiyi);
                matrix.scalarMultiply(w, (1 - (eta * alpha)), w1);
                matrix.add(w1, xiyi, w);
            } else {
                matrix.scalarMultiply(w, (1 - (eta * alpha)), w);
            }
            double end_compute = MPI_Wtime();
            compute_time += (end_compute - start_compute);
            //util.print1DMatrix(w, 5);
            cost = 0.5 * alpha * fabs(matrix.dot(w, w)) + max(0.0, (1 - yixiw));
            double start_cost = MPI_Wtime();
            local_cost[0] = cost;
            MPI_Allreduce(local_cost, global_cost, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            cost = global_cost[0] / (double) world_size;
            double end_cost = MPI_Wtime();
            predict_time += (end_cost - start_cost);
        }

        double start_communication = MPI_Wtime();
        MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        double end_communication = MPI_Wtime();
        communication_time += (end_communication - start_communication);
        double start_compute = MPI_Wtime();
        matrix.scalarMultiply(wglobal, 1.0 / (double) world_size, w);
        double end_compute = MPI_Wtime();
        compute_time += (end_compute - start_compute);


        if (iterations < i and world_rank == 0) {
            breakFlag[0] = -1;
        }
        double bcast_time_start = MPI_Wtime();
        MPI_Bcast(breakFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
//        if (breakFlag[0] == -1) {
//            cout << "World Rank : " << world_rank << "Break Flag : " << breakFlag[0] << endl;
//        }
        double bcast_time_end = MPI_Wtime();
        predict_time += (bcast_time_end - bcast_time_start);
        i++;
    }

    this->setTotalPredictionTime(predict_time);


    delete[] xiyi;
    delete[] w1;
}

void PSGD::pegasosSGDFullBatchv3(double *w, string epochlogfile) {
    Initializer initializer;
    double initTime = 0;
    double tempTime = 0;
    tempTime = MPI_Wtime();
    double *w1 = new double[features];
    initializer.initializeWeightsWithArray(features, w1);
    double *xiyi = new double[features];
    initializer.initializeWeightsWithArray(features, xiyi);
    double *wglobal = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal);
    double eta = 0;
    double communication_time = 0;
    double computation_time = 0;

    Util util;

    Matrix1 matrix(features);
    if (world_rank == 0) {
        cout << "No of Iterations : " << iterations << endl;
    }

    initializer.initialWeights(features, w);
    initTime = MPI_Wtime() - tempTime;
    int i = 1;
    double temp1 = 0;
    double cost = 0;
    double yixiw = 0;
    double t1 = MPI_Wtime();
    double invAlpha = 1.0 - alpha;
//    vec a = {1, 3, -5};
//    vec b = {4, -2, -1};
//    vec c = {};
//    std::vector<int> a1 = {1, 3, -5};
//    std::vector<int> b1 = {4, -2, -1};
//    int result = 0;
//
//    mat A = mat(2,4);
//    cout << A << endl;
//    A.ones();
//    cout << A << endl;
//    A.randu();
//    cout << A << endl;
//
//    mat aw = mat(7000,22);
//    mat bw = mat(1,22);
//
//    aw.randu();
//    bw.randu();


    alpha = -alpha;
    //vec d = vec()
    //vec B = randu(1,features);
    //vec W = randu(1,features);
    for (int k = 0; k < iterations; k++) {
        for (int j = 0; j < trainingSamples; ++j) {

            //yixiw = matrix.doti(X[j], w, yixiw) * y[j];
            yixiw = cblas_ddot(features, X[j], 1, w, 1);
            if (yixiw < 1) {
                //matrix.scalarMultiply(X[j], y[j] *  alpha, xiyi);
                //matrix.add(w1, xiyi, w);
                cblas_daxpy(features, alpha * y[j], X[j], 1, xiyi, 1.0);
                cblas_daxpy(features, alpha , xiyi, 1, w, 1.0);

            } else {
                //matrix.scalarMultiply(w, invAlpha, w);
                cblas_daxpy(features, alpha , w, 1, w, 1.0);
            }



            //result += a[i] * b[i];

        }
        temp1 = MPI_Wtime();
        MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        communication_time += MPI_Wtime() - temp1;

        matrix.scalarMultiply(wglobal, 1.0 / (double) world_size, w);

    }
    double t2 = MPI_Wtime();
    if (world_rank == 0) {
        cout << "Xtraining Time : " << t2 - t1 << " s" << endl;

        cout << "XTraining Samples : " << trainingSamples << endl;

    }


    this->setTotalPredictionTime(communication_time);
    this->setCompute_time(initTime);

    delete[] xiyi;
    delete[] w1;
    delete[] wglobal;
}

void PSGD::blassTest() {
    //Random numbers
    std::mt19937_64 rnd;
    std::uniform_real_distribution<double> doubleDist(0, 1);

    //Create Arrays that represent the matrices A,B,C
    const int n = 20;
    double*  A = new double[n*n];
    double*  B = new double[n*n];
    double*  C = new double[n*n];

    //Fill A and B with random numbers
    for(uint i =0; i <n; i++){
        for(uint j=0; j<n; j++){
            A[i*n+j] = doubleDist(rnd);
            B[i*n+j] = doubleDist(rnd);
        }
    }

    //Calculate A*B=C
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, B, n, 0.0, C, n);

    //Clean up
    delete[] A;
    delete[] B;
    delete[] C;
}

void PSGD::nonBlassTest() {
    cout << "Non Blass Test " << endl;
    std::mt19937_64 rnd;
    std::uniform_real_distribution<double> doubleDist(0, 1);

    //Create Arrays that represent the matrices A,B,C
    const int n = 20;
    double*  A = new double[n*n];
    double*  B = new double[n*n];
    double*  C = new double[n*n];

    //Fill A and B with random numbers
    for(uint i =0; i <n; i++){
        for(uint j=0; j<n; j++){
            A[i*n+j] = doubleDist(rnd);
            B[i*n+j] = doubleDist(rnd);
        }
    }

    //Calculate A*B=C
    //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, B, n, 0.0, C, n);
    for (int k = 0; k < n * n; ++k) {
        C[k] = A[k] * B[k];
    }

    //Clean up
    delete[] A;
    delete[] B;
    delete[] C;
}

void PSGD::pegasosSGDBatchv2(double *w, int comm_gap, string summarylogfile, string epochlogfile, string weightFile) {
    Initializer initializer;
    //cout << "Start Training ..." << endl;
    double init_time_start = MPI_Wtime();
    double *w1 = new double[features];
    initializer.initializeWeightsWithArray(features, w1);
    double *xiyi = new double[features];
    initializer.initializeWeightsWithArray(features, xiyi);
    double *wglobal = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal);
    double *wglobal_print = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal_print);
    double *w_print = new double[features];
    initializer.initializeWeightsWithArray(features, w_print);
    double *local_cost = new double[1];
    initializer.initializeWeightsWithArray(1, local_cost);
    double *global_cost = new double[1];
    initializer.initializeWeightsWithArray(1, global_cost);
    double *w_init = new double[features];
    initializer.initializeWeightsWithArray(features, w_init);


    double epsilon = 0.00000001;
    Util util;
    //cout << "Training Samples : " << trainingSamples << endl;
    //cout << "Beta 1 :" << beta1 << ", Beta 2 :" << beta2 << endl;
    //util.print1DMatrix(wInit, features);
    //util.print1DMatrix(v, features);
    //util.print1DMatrix(r, features);

    Matrix1 matrix(features);

    initializer.initialWeights(features, w);
    MPI_Allreduce(w, w_init, features, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    matrix.scalarMultiply(w_init, 1.0 / (double) world_size, w);


    compute_time = 0;
    communication_time = 0;
    double start_compute = 0;
    double end_compute = 0;
    double start_predict = 0;
    double end_predict = 0;
    double prediction_time = 0;
    double training_time = 0;
    double comm_time_calc = 0;
    double comp_time_calc = 0;
    double eta = 0;
    vector<int> accuracies_set(0);
    //shuffeling stage
    int indexes[trainingSamples];
//    if(world_rank==0) {
//
//        vector<int> indices(trainingSamples);
//        std::iota(indices.begin(), indices.end(), 0);
//        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
//
//        shuffle (indices.begin(), indices.end(), default_random_engine(seed));
//
//        cout << "Size of Indices : " << indices.size() <<  ", "<< trainingSamples << endl;
//
//        for (int k = 0; k < trainingSamples; ++k) {
//            indexes[k] = indices.at(k);
//        }
//    }
//
//
//    MPI_Bcast(indexes,trainingSamples, MPI_INT, 0, MPI_COMM_WORLD);

    vector<double> comptimeV;
    vector<double> commtimeV;
    double cost = 10.0;
    int breakFlag[] = {100};
    int i = 1;

    double error_threshold = this->getError_threshold();
    double error = 0;
    int marker = 0;
    double init_time_end = MPI_Wtime();
    double cost_sum = 0;
    double yixiw;
    double acc = 0;
    while (breakFlag[0] != -1) {
        eta = 1.0 / (alpha * i);
        for (int j = 0; j < trainingSamples; ++j) {
            double perDataPerItrCompt = 0;
            start_compute = MPI_Wtime();
            yixiw = matrix.dot(X[j], w);
            yixiw = yixiw * y[j];

            if (yixiw < 1) {
                matrix.scalarMultiply(X[j], y[j] * eta, xiyi);
                matrix.scalarMultiply(w, (1 - (eta * alpha)), w1);
                matrix.add(w1, xiyi, w);
            } else {
                matrix.scalarMultiply(w, (1 - (eta * alpha)), w);
            }

            end_compute = MPI_Wtime();
            compute_time += (end_compute - start_compute);
            perDataPerItrCompt = (end_compute - start_compute);
            if (j % comm_gap == 0) {
                double start_communication = MPI_Wtime();
                MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
                double end_communication = MPI_Wtime();
                communication_time += (end_communication - start_communication);
                start_compute = MPI_Wtime();
                matrix.scalarMultiply(wglobal, 1.0 / (double) world_size, w);
                end_compute = MPI_Wtime();
                compute_time += (end_compute - start_compute);
                perDataPerItrCompt += (end_compute - start_compute);
                comm_time_calc -= MPI_Wtime();
                commtimeV.push_back(end_communication - start_communication);
                comm_time_calc += MPI_Wtime();
            } else {
                commtimeV.push_back(0);
            }
            comp_time_calc -= MPI_Wtime();
            comptimeV.push_back(perDataPerItrCompt);
            comm_time_calc += MPI_Wtime();
            prediction_time += comm_time_calc + comp_time_calc;
            comm_time_calc = 0;
            comp_time_calc = 0;
        }
        training_time += communication_time + compute_time;
        cost = (0.5 * alpha * fabs(matrix.dot(w, w))) + max(0.0, (1 - yixiw));
        //double start_cost = MPI_Wtime();
        local_cost[0] = cost;
        MPI_Allreduce(local_cost, global_cost, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        cost = global_cost[0] / (double) world_size;

        start_predict = MPI_Wtime();
        MPI_Allreduce(w, wglobal_print, features, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        matrix.scalarMultiply(wglobal_print, 1.0 / (double) world_size, w_print);
        if (world_rank == 0) {
            Predict predict(Xtest, ytest, w_print, testingSamples, features);
            acc = predict.crossValidate();
            error = 100.0 - acc;
            cout << "Pegasos Batch PSGD Epoch : Rank : " << world_rank << ", Epoch " << i << "/" << iterations
                 << " Testing Accuracy : " << acc << "%" << ", Hinge Loss : " << cost << endl;
            util.writeTimeLossAccuracyPerEpoch(i, acc, cost, training_time, epochlogfile);
        }
        end_predict = MPI_Wtime();
        prediction_time += (end_predict - start_predict);

        double bcast_time_start = MPI_Wtime();


        if (cost < error_threshold and world_rank == 0) {
            accuracies_set.push_back(marker);
        } else {
            marker = 0;
            accuracies_set.clear();
        }

        if ((accuracies_set.size() == 5) or iterations < i) {
            breakFlag[0] = -1;
        }

        MPI_Bcast(breakFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
//        if(breakFlag[0]==-1){
//            cout << "World Rank : " << world_rank << "Break Flag : " << breakFlag[0] << endl;
//        }

        double bcast_time_end = MPI_Wtime();
        //prediction_time += (bcast_time_end-bcast_time_start);
        i++;
    }
    prediction_time += (init_time_end - init_time_start);


    double io_time = 0;
    io_time -= MPI_Wtime();
    string file = "";
    file.append(summarylogfile.append(util.getTimestamp()).append("_world_size=").append(to_string(world_size)).append(
            "_").append("_process=").append(to_string(world_rank)).append("_alpha_").append(to_string(alpha)).append(
            "_comm_gap=").append(to_string(comm_gap)));
    writeVectorLog(file, iterations, trainingSamples, comptimeV, commtimeV);
    io_time += MPI_Wtime();
    prediction_time += io_time; // prediction captures all time taken to io + cross-validation accuracy calculation + initialization time
    prediction_time = io_time; // overwrite with io_time to only see the effect from the cost-validation and initialization overhead

    this->setTotalPredictionTime(prediction_time);
    this->setError_threshold(error_threshold);
    this->setEffective_epochs(i);
    this->setResultant_final_cross_accuracy(acc);
    this->setResultant_minimum_cost(cost);

    delete[] w1;
    delete[] xiyi;
    delete[] wglobal;
    delete[] wglobal_print;
    delete[] w_print;
    delete[] local_cost;
    delete[] global_cost;

}


void PSGD::pegasosSGDBatchv2t1(double *w, int comm_gap, int threads, string summarylogfile, string epochlogfile,
                               string weightFile) {
    Initializer initializer;
    cout << "Start Training ..." << endl;
    double init_time_start = MPI_Wtime();
    double *w1 = new double[features];
    initializer.initializeWeightsWithArray(features, w1);
    double *xiyi = new double[features];
    initializer.initializeWeightsWithArray(features, xiyi);
    double *wglobal = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal);
    double *wglobal_print = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal_print);
    double *w_print = new double[features];
    initializer.initializeWeightsWithArray(features, w_print);
    double *local_cost = new double[1];
    initializer.initializeWeightsWithArray(1, local_cost);
    double *global_cost = new double[1];
    initializer.initializeWeightsWithArray(1, global_cost);
    double *w_init = new double[features];
    initializer.initializeWeightsWithArray(features, w_init);


    double epsilon = 0.00000001;
    Util util;
    //cout << "Training Samples : " << trainingSamples << endl;
    //cout << "Beta 1 :" << beta1 << ", Beta 2 :" << beta2 << endl;
    //util.print1DMatrix(wInit, features);
    //util.print1DMatrix(v, features);
    //util.print1DMatrix(r, features);

    Matrix1 matrix(features);

    initializer.initialWeights(features, w);
    MPI_Allreduce(w, w_init, features, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    matrix.scalarMultiply(w_init, 1.0 / (double) world_size, w);


    compute_time = 0;
    communication_time = 0;
    double start_compute = 0;
    double end_compute = 0;
    double start_predict = 0;
    double end_predict = 0;
    double prediction_time = 0;
    double training_time = 0;
    double comm_time_calc = 0;
    double comp_time_calc = 0;
    double eta = 0;
    vector<int> accuracies_set(0);
    //shuffeling stage
    int indexes[trainingSamples];
//    if(world_rank==0) {
//
//        vector<int> indices(trainingSamples);
//        std::iota(indices.begin(), indices.end(), 0);
//        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
//
//        shuffle (indices.begin(), indices.end(), default_random_engine(seed));
//
//        cout << "Size of Indices : " << indices.size() <<  ", "<< trainingSamples << endl;
//
//        for (int k = 0; k < trainingSamples; ++k) {
//            indexes[k] = indices.at(k);
//        }
//    }
//
//
//    MPI_Bcast(indexes,trainingSamples, MPI_INT, 0, MPI_COMM_WORLD);

    vector<double> comptimeV;
    vector<double> commtimeV;
    double cost = 10.0;
    int breakFlag[] = {100};
    int i = 1;

    double error_threshold = this->getError_threshold();
    double error = 0;
    int marker = 0;
    double init_time_end = MPI_Wtime();
    double cost_sum = 0;
    double yixiw;
    double acc = 0;
    int j = 0;
    for (i = 1; i < iterations; i++) {
        eta = 1.0 / (alpha * i);
        for (j = 0; j < trainingSamples; ++j) {

            double perDataPerItrCompt = 0;
            start_compute = MPI_Wtime();
            yixiw = matrix.dot(X[j], w);
            yixiw = yixiw * y[j];

            if (yixiw < 1) {
                int n1, n2, n3 = 0;
                int x = 10;


                //matrix.parallelScalarMultiply(X[j], y[j] * eta, xiyi);
#pragma omp parallel for
                for (int i1 = 0; i1 < features; ++i1) {
                    xiyi[i1] = X[j][i1] * (y[j] * eta);
                }

                //matrix.scalarMultiply(X[j], y[j] * eta, xiyi);
                matrix.scalarMultiply(w, (1 - (eta * alpha)), w1);
                matrix.add(w1, xiyi, w);
            } else {
                matrix.scalarMultiply(w, (1 - (eta * alpha)), w);
            }

            end_compute = MPI_Wtime();
            compute_time += (end_compute - start_compute);
            perDataPerItrCompt = (end_compute - start_compute);
            if (j % comm_gap == 0) {
                double start_communication = MPI_Wtime();
                MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
                double end_communication = MPI_Wtime();
                communication_time += (end_communication - start_communication);
                start_compute = MPI_Wtime();
                matrix.scalarMultiply(wglobal, 1.0 / (double) world_size, w);
                end_compute = MPI_Wtime();
                compute_time += (end_compute - start_compute);
                perDataPerItrCompt += (end_compute - start_compute);
                comm_time_calc -= MPI_Wtime();
                commtimeV.push_back(end_communication - start_communication);
                comm_time_calc += MPI_Wtime();
            } else {
                commtimeV.push_back(0);
            }
            comp_time_calc -= MPI_Wtime();
            comptimeV.push_back(perDataPerItrCompt);
            comm_time_calc += MPI_Wtime();
            prediction_time += comm_time_calc + comp_time_calc;
            comm_time_calc = 0;
            comp_time_calc = 0;
        }
        training_time += communication_time + compute_time;
        cost = (0.5 * alpha * fabs(matrix.dot(w, w))) + max(0.0, (1 - yixiw));
        //double start_cost = MPI_Wtime();
        local_cost[0] = cost;
        MPI_Allreduce(local_cost, global_cost, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        cost = global_cost[0] / (double) world_size;

        start_predict = MPI_Wtime();
        MPI_Allreduce(w, wglobal_print, features, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        matrix.scalarMultiply(wglobal_print, 1.0 / (double) world_size, w_print);
        if (world_rank == 0) {
            Predict predict(Xtest, ytest, w_print, testingSamples, features);
            acc = predict.predict();
            error = 100.0 - acc;
            cout << "Pegasos Batch PSGD Epoch : Rank : " << world_rank << ", Epoch " << i << "/" << iterations
                 << " Testing Accuracy : " << acc << "%" << ", Hinge Loss : " << cost << endl;
            util.writeTimeLossAccuracyPerEpoch(i, acc, cost, training_time, epochlogfile);
        }
        end_predict = MPI_Wtime();
        prediction_time += (end_predict - start_predict);
    }


    prediction_time += (init_time_end - init_time_start);


    double io_time = 0;
    io_time -= MPI_Wtime();
    string file = "";
    file.append(summarylogfile.append(util.getTimestamp()).append("_world_size=").append(to_string(world_size)).append(
            "_").append("_process=").append(to_string(world_rank)).append("_alpha_").append(to_string(alpha)).append(
            "_comm_gap=").append(to_string(comm_gap)));
    writeVectorLog(file, iterations, trainingSamples, comptimeV, commtimeV);
    io_time += MPI_Wtime();
    prediction_time += io_time;

    this->setTotalPredictionTime(prediction_time);
    this->setError_threshold(error_threshold);
    this->setEffective_epochs(i);
    this->setResultant_final_cross_accuracy(acc);
    this->setResultant_minimum_cost(cost);

    delete[] w1;
    delete[] xiyi;
    delete[] wglobal;
    delete[] wglobal_print;
    delete[] w_print;
    delete[] local_cost;
    delete[] global_cost;

}


void PSGD::adamSGDRotationv1(double *w) {
    Initializer initializer;
    Util util;
    if (world_rank == 0) {
        cout << "Start Training ..." << endl;
    }

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
    double *wglobal = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal);
    double *wcomm = new double[features];
    initializer.initializeWeightsWithArray(features, wcomm);
    double *wcomm1 = new double[features];
    initializer.initializeWeightsWithArray(features, wcomm1);
    double epsilon = 0.00000001;


    Matrix1 matrix(features);

    initializer.initialWeights(features, w);
    compute_time = 0;
    communication_time = 0;
    double start_compute = 0;
    double end_compute = 0;
    for (int i = 1; i < iterations; ++i) {
        // alpha = 1.0 / (1.0 + (double) i);
        if (i % 10 == 0 and world_rank == 0) {
            //cout << "+++++++++++++++++++++++++++++++++" << endl;
            //util.print1DMatrix(w, features);
            //cout << "+++++++++++++++++++++++++++++++++" << endl;
            cout << "Iteration " << i << "/" << iterations << endl;
        }
        for (int j = 0; j < trainingSamples; ++j) {
            start_compute = MPI_Wtime();
            double yixiw = matrix.dot(X[j], w);
            yixiw = yixiw * y[j];

            double coefficient = 1.00;//1.0 / (1.0 + double(i));

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
            end_compute = MPI_Wtime();
            compute_time += (end_compute - end_compute);


            int next = (world_rank + 1) % world_size;
            int prev = (world_rank + world_size - 1) % world_size;
            if (world_rank > 1) {
                double start_communication = MPI_Wtime();
                MPI_Send(w, features, MPI_DOUBLE, next, 1, MPI_COMM_WORLD);
                MPI_Recv(wcomm, features, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                double end_communication = MPI_Wtime();
                communication_time += (end_communication - start_communication);
                matrix.add(w, wcomm, wcomm1);
                matrix.scalarMultiply(wcomm1, 0.50, w);
            }

            //comms.send(w, dtype=comms.mpi.FLOAT,dest=next, tag=1)
            //w_next = comms.recv(dtype=comms.mpi.FLOAT, source=prev, tag=1, size=w.shape[0])
            //w = (w + w_next)/2.0
            //util.print1DMatrix(w, 8);
            //delete [] xi;
        }
    }
//    if (world_rank > 1) {
//        double start_final_comms = MPI_Wtime();
//        MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
//                      MPI_COMM_WORLD);
//        double end_final_comms = MPI_Wtime();
//        communication_time += (end_final_comms - start_final_comms);
//        start_compute = MPI_Wtime();
//        matrix.scalarMultiply(wglobal, 1.0 / (double) world_size, w);
//        end_compute = MPI_Wtime();
//        compute_time += (end_compute - start_compute);
//    }

    /*if(world_rank==0) {
        cout << "============================================" << endl;
        printf("Final Weight\n");
        util.print1DMatrix(w, features);

        cout << "============================================" << endl;
    }*/

    //cout << "Compute Time of Rank : " << world_rank << " is " << compute_time << endl;
    //cout << "Communication Time of Rank : " << world_rank << " is " << communication_time << endl;

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
    delete[] wcomm;
    delete[] wcomm1;

}

void PSGD::adamSGDRandomRingv1(double *w, double dropout_per, string logfile) {
    int *active_ranks;
    int miss_rank_size = dropout_per * world_size;
    int active_rank_size = world_size - miss_rank_size;
    active_ranks = new int[active_rank_size];
    if (world_rank == 0) {
        generateRandomRanks(active_ranks, dropout_per);
    }
    MPI_Bcast(active_ranks, active_rank_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    string s;
    s.append("Rank : ").append(to_string(world_rank)).append(", ");
    for (int i = 0; i < active_rank_size; ++i) {
        s.append(to_string(active_ranks[i])).append(" ");
    }
    if (world_rank == 0) {
        cout << s << endl;
    }


    Initializer initializer;
    if (world_rank == 0) {
        cout << "Start Training ..." << endl;
    }

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
    double *wglobal = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal);


    double epsilon = 0.00000001;


    Matrix1 matrix(features);

    initializer.initialWeights(features, w);
    compute_time = 0;
    communication_time = 0;
    double start_compute = 0;
    double end_compute = 0;
    double cost = 0;
    for (int i = 1; i < iterations; ++i) {
        if (i % 10 == 0 and world_rank == 0) {
            //cout << "+++++++++++++++++++++++++++++++++" << endl;
            //util.print1DMatrix(w, features);
            //cout << "+++++++++++++++++++++++++++++++++" << endl;
            cout << "Iteration " << i << "/" << iterations << ", Cost : " << cost << endl;
        }
        for (int j = 0; j < trainingSamples; ++j) {
            start_compute = MPI_Wtime();
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
            end_compute = MPI_Wtime();
            compute_time += (end_compute - end_compute);


            if (isIncluded(active_ranks, world_rank, active_rank_size)) {
                int my_index = getRankIndex(active_ranks, world_rank, active_rank_size);
                int nextId = 0; //(world_rank + 1) % active_rank_size;
                int prevId = 0; //(world_rank + active_rank_size - 1) % active_rank_size;
                if (my_index - 1 >= 0) {
                    prevId = my_index - 1;
                }

                if (my_index - 1 < 0) {
                    prevId = active_rank_size - 1;
                }
                if ((my_index + 1) < (active_rank_size)) {
                    nextId = my_index + 1;
                }
                if ((my_index + 1) == active_rank_size) {
                    nextId = 0;
                }

//                cout << "My Rank : " << world_rank << ", aPrevious Rank : " << active_ranks[prevId] << ", Next Rank : "
//                     << active_ranks[nextId] << endl;

                int next = active_ranks[nextId];
                int prev = active_ranks[prevId];
                if (world_rank > 1) {
                    double start_communication = MPI_Wtime();
                    MPI_Send(w, features, MPI_DOUBLE, next, 1, MPI_COMM_WORLD);
                    MPI_Recv(wglobal, features, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    double end_communication = MPI_Wtime();
                    communication_time += (end_communication - start_communication);
                    matrix.add(w, wglobal, w);
                    matrix.scalarMultiply(w, 1.0 / 2.0, w);
                }
            }
            //cost = abs(0.5 * np.dot(w, w.T) + self.C * condition)
            double c1 = matrix.dot(w, w);
            double c2 = 1 * yixiw;
            cost = abs(c1 + c2);


            //comms.send(w, dtype=comms.mpi.FLOAT,dest=next, tag=1)
            //w_next = comms.recv(dtype=comms.mpi.FLOAT, source=prev, tag=1, size=w.shape[0])
            //w = (w + w_next)/2.0
            //util.print1DMatrix(w, features);
            //delete [] xi;
        }
    }
    if (world_rank > 1) {
        double start_final_comms = MPI_Wtime();
        MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        double end_final_comms = MPI_Wtime();
        communication_time += (end_final_comms - start_final_comms);
        start_compute = MPI_Wtime();
        matrix.scalarMultiply(wglobal, 1.0 / (double) world_size, w);
        end_compute = MPI_Wtime();
        compute_time += (end_compute - start_compute);
    }

    /*if(world_rank==0) {
        cout << "============================================" << endl;
        printf("Final Weight\n");
        util.print1DMatrix(w, features);

        cout << "============================================" << endl;
    }*/

    //cout << "Compute Time of Rank : " << world_rank << " is " << compute_time << endl;
    //cout << "Communication Time of Rank : " << world_rank << " is " << communication_time << endl;
    if (world_rank == 0) {
        cout << "Training Completed ..." << endl;
    }
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

void PSGD::adamSGDRandomRingv2(double *w, double dropout_per, string logfile) {
    int *active_ranks;
    int miss_rank_size = dropout_per * world_size;
    int active_rank_size = world_size - miss_rank_size;
    active_ranks = new int[active_rank_size];


    Initializer initializer;
    if (world_rank == 0) {
        cout << "Start Training RandomRing v2 ..." << endl;
    }

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
    double *wglobal = new double[features];
    initializer.initializeWeightsWithArray(features, wglobal);
    double epsilon = 0.00000001;


    Matrix1 matrix(features);

    initializer.initializeWeightsWithArray(features, w);
    compute_time = 0;
    communication_time = 0;
    double start_compute = 0;
    double end_compute = 0;
    for (int i = 1; i < iterations; ++i) {
        if (world_rank == 0) {
            generateRandomRanks(active_ranks, dropout_per);
        }
        MPI_Bcast(active_ranks, active_rank_size, MPI_INT, 0, MPI_COMM_WORLD);
        //MPI_Barrier(MPI_COMM_WORLD);
//        string s;
//        s.append("Rank : ").append(to_string(world_rank)).append(", ");
//        for (int i = 0; i < active_rank_size; ++i) {
//            s.append(to_string(active_ranks[i])).append(" ");
//        }
//        if (world_rank == 0) {
//            cout << s << endl;
//        }
//        if (i % 1 == 0 and world_rank==0) {
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            //util.print1DMatrix(w, features);
//            //cout << "+++++++++++++++++++++++++++++++++" << endl;
//            cout << "Iteration " << i << "/" << iterations << endl;
//        }
        for (int j = 0; j < trainingSamples; ++j) {
            start_compute = MPI_Wtime();
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
            end_compute = MPI_Wtime();
            compute_time += (end_compute - end_compute);


            if (isIncluded(active_ranks, world_rank, active_rank_size)) {
                int my_index = getRankIndex(active_ranks, world_rank, active_rank_size);
                int nextId = 0; //(world_rank + 1) % active_rank_size;
                int prevId = 0; //(world_rank + active_rank_size - 1) % active_rank_size;
                if (my_index - 1 >= 0) {
                    prevId = my_index - 1;
                }

                if (my_index - 1 < 0) {
                    prevId = active_rank_size - 1;
                }
                if ((my_index + 1) < (active_rank_size)) {
                    nextId = my_index + 1;
                }
                if ((my_index + 1) == active_rank_size) {
                    nextId = 0;
                }

//                cout << "My Rank : " << world_rank << ", aPrevious Rank : " << active_ranks[prevId] << ", Next Rank : "
//                     << active_ranks[nextId] << endl;

                int next = active_ranks[nextId];
                int prev = active_ranks[prevId];
                if (world_rank > 1) {
                    double start_communication = MPI_Wtime();
                    MPI_Send(w, features, MPI_DOUBLE, next, 1, MPI_COMM_WORLD);
                    MPI_Recv(wglobal, features, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    double end_communication = MPI_Wtime();
                    communication_time += (end_communication - start_communication);
                    matrix.add(w, wglobal, w);
                    matrix.scalarMultiply(w, 1.0 / 2.0, w);
                }
            }




            //comms.send(w, dtype=comms.mpi.FLOAT,dest=next, tag=1)
            //w_next = comms.recv(dtype=comms.mpi.FLOAT, source=prev, tag=1, size=w.shape[0])
            //w = (w + w_next)/2.0
            //util.print1DMatrix(w, features);
            //delete [] xi;
        }
    }
    if (world_rank > 1) {
        double start_final_comms = MPI_Wtime();
        MPI_Allreduce(w, wglobal, features, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        double end_final_comms = MPI_Wtime();
        communication_time += (end_final_comms - start_final_comms);
        start_compute = MPI_Wtime();
        matrix.scalarMultiply(wglobal, 1.0 / (double) world_size, w);
        end_compute = MPI_Wtime();
        compute_time += (end_compute - start_compute);
    }

    /*if(world_rank==0) {
        cout << "============================================" << endl;
        printf("Final Weight\n");
        util.print1DMatrix(w, features);

        cout << "============================================" << endl;
    }*/

    //cout << "Compute Time of Rank : " << world_rank << " is " << compute_time << endl;
    //cout << "Communication Time of Rank : " << world_rank << " is " << communication_time << endl;
    if (world_rank == 0) {
        cout << "Training Completed ..." << endl;
    }
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


int PSGD::getRankIndex(int *active_ranks, int my_rank, int size) {
    int id = -1;
    for (int i = 0; i < size; ++i) {
        if (active_ranks[i] == my_rank) {
            id = i;
            break;
        }
    }
    return id;
}


bool PSGD::isIncluded(int *active_ranks, int my_rank, int size) {
    bool status = false;
    for (int i = 0; i < size; ++i) {
        if (active_ranks[i] == my_rank) {
            status = true;
            break;
        }
    }
    return status;
}

void PSGD::generateRandomRanks(int *active_ranks, double dropout_per) {
    int miss_rank_size = dropout_per * world_size;
    int active_rank_size = world_size - miss_rank_size;
    mt19937 rng;
    rng.seed(random_device()());
    uniform_int_distribution<mt19937::result_type> dist6(0, world_size - 1);
    //cout << "Drop out Rank size : " << miss_rank_size << endl;
    //cout << "Active Rank Size : " << active_rank_size << endl;


    for (int i = 0; i < active_rank_size; ++i) {
        active_ranks[i] = -1;
    }
    int j = 0;
    while (true) {
        int random_index = dist6(rng);

        if (!isPresent(active_ranks, random_index, active_rank_size)) {
            active_ranks[j] = random_index;
            j++;
        }

        if (isPossibleRanks(active_ranks, active_rank_size)) {
            break;
        }

    }
//    for (int i = 0; i < active_rank_size; ++i) {
//        cout << active_ranks[i] << " ";
//    }
//    cout << endl;

}

bool PSGD::isPresent(int *arr, int new_rank, int size) {
    bool isPresent = false;
    for (int i = 0; i < size; ++i) {
        if (arr[i] == new_rank) {
            isPresent = true;
            break;
        }
    }
    return isPresent;
}

bool PSGD::isPossibleRanks(int *arr, int size) {
    int count = 0;
    for (int i = 0; i < size; ++i) {
        if (arr[i] != -1) {
            count++;
        }
    }
    if (count == size) {
        return true;
    } else {
        return false;
    }
}

void PSGD::writeLog(string logfile, int iterations, int samples, double **compt, double **commt) {
    cout << logfile << commt[0][0] << "," << compt[0][0] << endl;
    ofstream myfile(logfile);
    if (myfile.is_open()) {
        for (int i = 1; i < iterations; ++i) {
            double per_itr_comp = 0;
            double per_itr_comm = 0;
            for (int j = 0; j < samples; ++j) {
                per_itr_comm += commt[i][j];
                per_itr_comp += compt[i][j];
            }
            cout << i << "," << per_itr_comp << "," << per_itr_comm << "\n";
            myfile << i << "," << per_itr_comp << "," << per_itr_comm << "\n";
        }
        myfile.close();
    }
}

void PSGD::writeVectorLog(string logfile, int iterations, int samples, vector<double> compt, vector<double> commt) {
    //cout << logfile << endl;
    ofstream myfile(logfile);
    if (myfile.is_open()) {
        double per_itr_comp = 0;
        double per_itr_comm = 0;
        for (int i = 0; i < compt.size(); ++i) {
            per_itr_comm += commt.at(i);
            per_itr_comp += compt.at(i);
            if ((i + 1) % trainingSamples == 0) {
                //cout << i << "," << per_itr_comp << "," << per_itr_comm << "\n";
                myfile << i / trainingSamples << "," << per_itr_comp << "," << per_itr_comm << "\n";
                per_itr_comm = 0;
                per_itr_comp = 0;
            }
        }
        myfile.close();
    }
}

void PSGD::writeLog(string logfile, int iterations, int samples, double **compt, double *commt) {
    cout << logfile << commt[0] << "," << compt[0][0] << endl;
    ofstream myfile(logfile);
    if (myfile.is_open()) {
        for (int i = 1; i < iterations; ++i) {
            double per_itr_comp = 0;
            double per_itr_comm = 0;
            per_itr_comm += commt[i];
            for (int j = 0; j < samples; ++j) {

                per_itr_comp += compt[i][j];
            }
            cout << i << "," << per_itr_comp << "," << per_itr_comm << "\n";
            myfile << i << "," << per_itr_comp << "," << per_itr_comm << "\n";
        }
        myfile.close();
    }
}

void PSGD::writeLog(string logfile, int iterations, int samples, double **compt, double commt) {
    cout << logfile << commt << "," << compt[0][0] << endl;
    ofstream myfile(logfile);
    if (myfile.is_open()) {
        for (int i = 1; i < iterations; ++i) {
            double per_itr_comp = 0;
            double per_itr_comm = commt;
            for (int j = 0; j < samples; ++j) {

                per_itr_comp += compt[i][j];
            }
            //cout << i << "," << per_itr_comp << "," << per_itr_comm << "\n";
            myfile << i << "," << per_itr_comp << "," << per_itr_comm << "\n";
        }
        myfile.close();
    }
}

double *PSGD::getW() const {
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

double PSGD::getCompute_time() const {
    return compute_time;
}

void PSGD::setCompute_time(double compute_time) {
    PSGD::compute_time = compute_time;
}

double PSGD::getCommunication_time() const {
    return communication_time;
}

void PSGD::setCommunication_time(double communication_time) {
    PSGD::communication_time = communication_time;
}

const vector<double> &PSGD::getCompute_time_of_ranks() const {
    return compute_time_of_ranks;
}

const vector<double> &PSGD::getCommunication_time_of_ranks() const {
    return communication_time_of_ranks;
}

PSGD::PSGD(double beta1, double beta2, double **X, double *y, double alpha, int iterations, int features,
           int trainingSamples, int testingSamples, int world_size, int world_rank, double **Xtest, double *ytest)
        : beta1(beta1), beta2(beta2), X(X), y(y), alpha(alpha), iterations(iterations), features(features),
          trainingSamples(trainingSamples), testingSamples(testingSamples), world_size(world_size),
          world_rank(world_rank), Xtest(Xtest), ytest(ytest) {}

double PSGD::getTotalPredictionTime() const {
    return totalPredictionTime;
}

void PSGD::setTotalPredictionTime(double totalPredictionTime) {
    PSGD::totalPredictionTime = totalPredictionTime;
}

double PSGD::getError_threshold() const {
    return error_threshold;
}

void PSGD::setError_threshold(double error_threshold) {
    PSGD::error_threshold = error_threshold;
}

int PSGD::getEffective_epochs() const {
    return effective_epochs;
}

void PSGD::setEffective_epochs(int effective_epochs) {
    PSGD::effective_epochs = effective_epochs;
}

double PSGD::getResultant_minimum_cost() const {
    return resultant_minimum_cost;
}

void PSGD::setResultant_minimum_cost(double resultant_minimum_cost) {
    PSGD::resultant_minimum_cost = resultant_minimum_cost;
}

double PSGD::getResultant_final_cross_accuracy() const {
    return resultant_final_cross_accuracy;
}

void PSGD::setResultant_final_cross_accuracy(double resultant_final_cross_accuracy) {
    PSGD::resultant_final_cross_accuracy = resultant_final_cross_accuracy;
}


