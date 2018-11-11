//
// Created by vibhatha on 11/5/18.
//
#include <iostream>
#include <cmath>
#include "Test.h"
#include "Util.h"
#include "DataSet.h"
#include "ResourceManager.h"

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