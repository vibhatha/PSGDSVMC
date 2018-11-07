#include <iostream>
#include "Test.h"
#include "DataSet.h"
#include "Util.h"
#include "SGD.h"
#include <ctime>

using namespace std;

void test1();
void test2();
void sgd();


int main() {
    std::cout << "Hello, World!" << std::endl;

    sgd();

    return 0;
}

void sgd() {
    string datasourceBase = "/home/vibhatha/data/svm/";
    string datasource = "heart";
    string fileName = "/training.csv";
    string sourceFile;
    sourceFile.append(datasourceBase).append(datasource).append(fileName);
    int features = 13;
    int trainingSamples = 216;
    int testingSamples = 64;

    DataSet dataset(sourceFile, features, trainingSamples, testingSamples);
    dataset.load();
    double** Xtrain = dataset.getXtrain();
    double* ytrain = dataset.getYtrain();
    Util util;
    //util.print2DMatrix(Xtrain, trainingSamples, features);
    clock_t begin = clock();
    SGD sgd1(Xtrain, ytrain, 0.01, 200, features, trainingSamples, testingSamples);
    sgd1.sgd();
    clock_t end = clock();
    double elapsed_secs = double((end - begin) / double(CLOCKS_PER_SEC));
    printf("Time taken %f s", elapsed_secs);

}

void test1(){
    Test test;

    test.test1();
}

void test2() {
    Test test;
    test.test2();
}