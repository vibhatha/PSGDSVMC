#include <iostream>
#include "Test.h"
#include "DataSet.h"
#include "Util.h"
#include "SGD.h"

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
    double* ytrain = dataset.getYtrain();
    Util util;
    //util.printX(Xtrain, trainingSamples, features);
    SGD sgd1(Xtrain, ytrain, 0.01, 200);
    sgd1.sgd();

}

void test1(){
    Test test;

    test.test1();
}

void test2() {
    Test test;
    test.test2();
}