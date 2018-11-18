//
// Created by vibhatha on 11/5/18.
//
#include <iostream>
#include <cmath>
#include "Test.h"
#include "Util.h"
#include "DataSet.h"
#include "ResourceManager.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <time.h>
#include <ctime>
#include "Initializer.h"
#include "Matrix.h"

using namespace std;

Test::Test() {

}

void Test::test1() {
    double** a = new double*[10];

    for (int i = 0; i < 10; ++i) {
        a[i] = new double[10];
    }

    for (int j = 0; j < 10; ++j) {
        for (int i = 0; i < 10; ++i) {
            a[i][j] = i*j*1.0;
        }
    }

    for (int j = 0; j < 10; ++j) {
        for (int i = 0; i < 10; ++i) {
            printf("%f ",a[i][j]);
        }
        printf("\n");
    }

    Util util;
    util.print2DMatrix(a, 10, 10);
}

void Test::test2() {
    printf("Test 2 \n");
    string datasourceBase = "home/vibhatha/data/svm/";
    string datasource = "a9a";
    string fileName = "/training_mini.csv";
    string sourceFile;
    sourceFile.append(datasourceBase).append(datasource).append(fileName);
    int features = 123;
    int trainingSamples = 32561;
    int testingSamples = 16281;

    DataSet dataset(sourceFile, features, trainingSamples, testingSamples);
    dataset.load();
    double** Xtrain = dataset.getXtrain();
    Util util;
    util.print2DMatrix(Xtrain, trainingSamples, features);

}

void Test::test3() {
    double d = (pow(0.5,(double)10));
    cout <<" d: " << d << endl;
}

void Test::test4() {
    ResourceManager resourceManager;
    resourceManager.loadDataSourcePath();
    cout << resourceManager.getDataSourceBasePath() << endl;

}

void Test::test5() {
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    Initializer initializer;
    double* localsum = initializer.zeroWeights(2);
    double* globalsum = initializer.zeroWeights(2);


    if(world_rank % 2 == 1)
    {
        localsum[0] += 5;
    }
    else if( world_rank > 0 && (world_rank % 2 == 0))
    {
        localsum[1] += 10;
    }

    MPI_Allreduce(localsum, globalsum, 2, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    cout << "World Size : " << world_size << ", World Rank : " << world_rank << endl;
    Util util;
    util.print1DMatrix(globalsum,2);
    MPI_Finalize();
}

void Test::test6() {
    Util util;
    int f = 3;
    double* a = new double[f];
    double* b = new double[f];
    a[0] = 1.0;
    b[0]=2.0;
    a[1] = 2.0;
    b[1]=3.0;
    a[2] = 4.0;
    b[2]=5.0;
    Matrix matrix(f);
    double* res = new double[f];
    for (int i = 0; i < 10 ; ++i) {
        res = matrix.subtract(matrix.scalarMultiply(b,0.01, res ), b, res);
        util.print1DMatrix(res, f);

    }
}

void Test::test7() {
    std::time_t t = std::time(0);   // get time now
    std::tm* now = std::localtime(&t);
    std::cout << (now->tm_year + 1900) << '-'
              << (now->tm_mon + 1) << '-'
              <<  now->tm_mday
              << "\n";



}

void Test::test8() {

}

