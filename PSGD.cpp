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

using namespace std;

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
        Predict predict(Xtest, ytest, w , testingSamples, features);
        double acc = predict.predict();
        cout << "PSGD Epoch " << i << " Testing Accuracy : " << acc << "%" << endl;
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
        if(world_rank==0) {
            util.print1DMatrix(w, 5);
            Predict predict(Xtest, ytest, w , testingSamples, features);
            double acc = predict.predict();
            cout << "PSGD Batch v2 Epoch " << i << " Testing Accuracy : " << acc << "%" << endl;
            util.writeAccuracyPerEpoch(i, acc, epochlogfile);
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



    delete[] gradient;
    delete[] w_xiyi;
    delete[] aw1;
    delete[] xiyi;
    delete[] commtimeA;
    delete[] comptimeA;
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
        if (i % 10 == 0 and world_rank==0) {
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
    if(world_rank==0) {
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
        if (i % 10 == 0 and world_rank==0) {
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
                if(world_rank>1) {
                    double start_communication = MPI_Wtime();
                    MPI_Send(w, features, MPI_DOUBLE, next, 1, MPI_COMM_WORLD);
                    MPI_Recv(wglobal, features, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    double end_communication = MPI_Wtime();
                    communication_time += (end_communication - start_communication);
                    matrix.add(w,wglobal,w);
                    matrix.scalarMultiply(w, 1.0/2.0, w);
                }
            }
            //cost = abs(0.5 * np.dot(w, w.T) + self.C * condition)
            double c1 = matrix.dot(w,w);
            double c2 = 1 * yixiw;
            cost = abs(c1+c2);


            //comms.send(w, dtype=comms.mpi.FLOAT,dest=next, tag=1)
            //w_next = comms.recv(dtype=comms.mpi.FLOAT, source=prev, tag=1, size=w.shape[0])
            //w = (w + w_next)/2.0
            //util.print1DMatrix(w, features);
            //delete [] xi;
        }
    }
    if(world_rank>1) {
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
    if(world_rank==0) {
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
    if(world_rank==0) {
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
                if(world_rank>1) {
                    double start_communication = MPI_Wtime();
                    MPI_Send(w, features, MPI_DOUBLE, next, 1, MPI_COMM_WORLD);
                    MPI_Recv(wglobal, features, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    double end_communication = MPI_Wtime();
                    communication_time += (end_communication - start_communication);
                    matrix.add(w,wglobal,w);
                    matrix.scalarMultiply(w, 1.0/2.0, w);
                }
            }




            //comms.send(w, dtype=comms.mpi.FLOAT,dest=next, tag=1)
            //w_next = comms.recv(dtype=comms.mpi.FLOAT, source=prev, tag=1, size=w.shape[0])
            //w = (w + w_next)/2.0
            //util.print1DMatrix(w, features);
            //delete [] xi;
        }
    }
    if(world_rank>1) {
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
    if(world_rank==0) {
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
            //cout << i << "," << per_itr_comp << "," << per_itr_comm << "\n";
            myfile << i << "," << per_itr_comp << "," << per_itr_comm << "\n";
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


