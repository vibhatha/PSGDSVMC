#include <iostream>
#include "Test.h"
#include "DataSet.h"
#include "Util.h"
#include "SGD.h"
#include "ArgReader.h"
#include "Predict.h"
#include "ResourceManager.h"
#include <ctime>

using namespace std;

void test1();
void test2();
void test3();
void test4();
void sgd();
void train(OptArgs optArgs);


int main(int argc, char** argv) {
    std::cout << "Hello, World!" << std::endl;

    ArgReader argReader(argc, argv);
    OptArgs optArgs = argReader.getParams();
    train(optArgs);
    //sgd();
    //test4();

    return 0;
}

void train(OptArgs optArgs) {
    optArgs.toString();
    ResourceManager resourceManager;
    resourceManager.loadDataSourcePath();
    if(optArgs.isIsSplit()){
        string datasourceBase = resourceManager.getDataSourceBasePath();
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
        //util.print2DMatrix(Xtrain, trainSet, features);
        printf("\n----------------------------------------\n");
        //util.print2DMatrix(Xtest, testSet, features);
        clock_t begin = clock();
        SGD sgd1(Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, trainSet, testSet);
        SGD sgd2(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, trainSet);
        sgd2.adamSGD();
        //sgd1.sgd();
        clock_t end = clock();
        double elapsed_secs = double((end - begin) / double(CLOCKS_PER_SEC));
        printf("Training Samples : % d \n", trainSet);
        printf("Testing Samples : % d \n", testSet);
        printf("Training time %f s \n", elapsed_secs);
        //double* wFinal = sgd2.getWFinal();
        //util.print1DMatrix(wFinal, features);
//        double  wFinalTest [22] = {1.058225086609490377e-02,
//                2.280953604657339449e-03,
//                -1.943556791991714111e-05,
//                -1.077843995538567932e-02,
//                -8.031186202692977907e-03,
//                5.604057416423767826e-04,
//                -3.297580529278739542e-02,
//                -7.734271973922541600e-04,
//                -5.926790741002434248e-03,
//                -7.005303267160217610e-03,
//                5.329900037406029023e-01,
//                -5.146574347588905418e+00,
//                1.647242436140160526e-01,
//                8.668693932380384937e-02,
//                6.062124153509106800e-02,
//                8.558154902527673191e-02,
//                -2.200800258968939604e-01,
//                -3.348020131636486596e-01,
//                -1.974553008397789966e-01,
//                1.495450499343256301e-01,
//                1.682656341270570011e-01,
//                1.148141532462719216e-01};
//        Predict predict(Xtest, ytest, wFinalTest , testSet, features);
//        double acc = predict.predict();
//        cout << "Testing Accuracy : " << acc << "%" << endl;


    }else{
        string datasourceBase = resourceManager.getDataSourceBasePath();
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
        Util util;
//        util.print2DMatrix(Xtrain, trainingSamples, features);
        printf("\n----------------------------------------\n");
//        util.print2DMatrix(Xtest, testingSamples, features);
        clock_t begin = clock();
        SGD sgd1(Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, trainingSamples, testingSamples);
        SGD sgd2(0.5, 0.5, Xtrain, ytrain, optArgs.getAlpha(), optArgs.getIterations(), features, trainingSamples);
        sgd2.adamSGD();
        //sgd1.sgd();
        clock_t end = clock();
        double elapsed_secs = double((end - begin) / double(CLOCKS_PER_SEC));
        printf("Training Samples : % d \n", trainingSamples);
        printf("Testing Samples : % d \n", testingSamples);
        printf("Training time %f s \n", elapsed_secs);
        double* wFinal = sgd2.getWFinal();
        util.print1DMatrix(wFinal, features);
//        double  wFinalTest [22] = {1.058225086609490377e-02,
//                                   2.280953604657339449e-03,
//                                   -1.943556791991714111e-05,
//                                   -1.077843995538567932e-02,
//                                   -8.031186202692977907e-03,
//                                   5.604057416423767826e-04,
//                                   -3.297580529278739542e-02,
//                                   -7.734271973922541600e-04,
//                                   -5.926790741002434248e-03,
//                                   -7.005303267160217610e-03,
//                                   5.329900037406029023e-01,
//                                   -5.146574347588905418e+00,
//                                   1.647242436140160526e-01,
//                                   8.668693932380384937e-02,
//                                   6.062124153509106800e-02,
//                                   8.558154902527673191e-02,
//                                   -2.200800258968939604e-01,
//                                   -3.348020131636486596e-01,
//                                   -1.974553008397789966e-01,
//                                   1.495450499343256301e-01,
//                                   1.682656341270570011e-01,
//                                   1.148141532462719216e-01};
        Predict predict(Xtest, ytest, wFinal , testingSamples, features);
        double acc = predict.predict();
        cout << "Testing Accuracy : " << acc << "%" << endl;
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

void test3() {
    Test test;
    test.test3();
}

void test4() {
    Test test;
    test.test4();
}