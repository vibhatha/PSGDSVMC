#include <iostream>
#include "Test.h"
#include "DataSet.h"
#include "Util.h"
#include "SGD.h"
#include "ArgReader.h"
#include "Predict.h"
#include <ctime>

using namespace std;

void test1();
void test2();
void sgd();
void train(OptArgs optArgs);


int main(int argc, char** argv) {
    std::cout << "Hello, World!" << std::endl;

    ArgReader argReader(argc, argv);
    OptArgs optArgs = argReader.getParams();
    train(optArgs);
    //sgd();

    return 0;
}

void train(OptArgs optArgs) {
    optArgs.toString();
    if(optArgs.isIsSplit()){
        string datasourceBase = "/home/vibhatha/data/svm/";
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string sourceFile;
        sourceFile.append(datasourceBase).append(datasource).append(trainFileName);
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        double ratio = optArgs.getRatio();
        DataSet dataSet(sourceFile, features, trainingSamples, optArgs.isIsSplit(), ratio);
        dataSet.load();

        double** Xtrain = dataSet.getXtrain();
        double* ytrain = dataSet.getYtrain();

        double** Xtest = dataSet.getXtest();
        double* ytest = dataSet.getYtest();
        int totalSamples = trainingSamples;
        int trainSet = totalSamples * ratio;
        int testSet = totalSamples - trainSet;
        Util util;
        util.print2DMatrix(Xtrain, trainSet, features);
        printf("\n----------------------------------------\n");
        util.print2DMatrix(Xtest, testSet, features);
        clock_t begin = clock();
        SGD sgd1(Xtrain, ytrain, 0.01, 200, features, trainSet, testSet);
        sgd1.sgd();
        clock_t end = clock();
        double elapsed_secs = double((end - begin) / double(CLOCKS_PER_SEC));
        printf("Training Samples : % d \n", trainSet);
        printf("Testing Samples : % d \n", testSet);
        printf("Training time %f s \n", elapsed_secs);
        double* wFinal = sgd1.getW();
        util.print1DMatrix(wFinal, features);
        Predict predict(Xtest, ytest, wFinal , testSet, features);
        double acc = predict.predict();
        printf("Testing Accuracy %f % \n", acc);


    }else{
        string datasourceBase = "/home/vibhatha/data/svm/";
        string datasource = optArgs.getDataset();
        string trainFileName = "/training.csv";
        string testFileName = "/testing.csv";
        string trainFilePath;
        trainFilePath.append(datasourceBase).append(datasource).append(trainFileName);
        string testFilePath;
        testFilePath.append(datasourceBase).append(datasource).append(testFileName);
        int features = optArgs.getFeatures();
        int trainingSamples = optArgs.getTrainingSamples();
        int testingSamples = optArgs.getTestingSamples();

        DataSet dataset(features, trainingSamples, testingSamples, trainFilePath, testFilePath);
        dataset.load();
        double** Xtrain = dataset.getXtrain();
        double* ytrain = dataset.getYtrain();

        double** Xtest = dataset.getXtest();
        double* ytest = dataset.getYtest();
//        Util util;
//        util.print2DMatrix(Xtrain, trainingSamples, features);
//        printf("\n----------------------------------------\n");
//        util.print2DMatrix(Xtest, testingSamples, features);

        clock_t begin = clock();
        SGD sgd1(Xtrain, ytrain, 0.01, 200, features, trainingSamples, testingSamples);
        sgd1.sgd();
        clock_t end = clock();
        double elapsed_secs = double((end - begin) / double(CLOCKS_PER_SEC));
        printf("Training time %f s \n", elapsed_secs);
        Predict predict(Xtest, ytest, sgd1.getW(), testingSamples, features);
        double acc = predict.predict();
        printf("Testing Accuracy %f % \n", acc);
    }
}

void sgd() {
    string datasourceBase = "/home/vibhatha/data/svm/";
    string datasource = "heart";
    string trainFileName = "/training.csv";
    string testFileName = "/testing.csv";
    string sourceFile;
    sourceFile.append(datasourceBase).append(datasource).append(trainFileName);
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