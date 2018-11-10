//
// Created by vibhatha on 11/5/18.
//
#include <iostream>
#ifndef PSGDC_DATASET_H
#define PSGDC_DATASET_H

using namespace std;

class DataSet {
private:
    string sourceFile;
    int features;
    int trainingSamples;
    int testingSamples;
    bool isSplit;
    double ratio;
    double** Xtrain;
    double* ytrain;
    double** Xtest;
    double* ytest;
    string trainFile;
    string testFile;

public:
    DataSet(int features, int trainingSamples, int testingSamples, const string &trainFile, const string &testFile);
    DataSet(string sourceFile_, int features_, int trainingSamples_, int testingSamples_);
    DataSet(string sourceFile_,  int features_, int trainingSamples_, bool isSplit_, double ratio_);
    DataSet(string sourceFile_,  int features_, int trainingSamples_, int testingSamples_, bool isSplit_, double ratio_);
    void load();
    double** getXtrain();
    double** getXtest();
    double* getYtrain();
    double* getYtest();

    int getTrainingSamples() const;

    void setTrainingSamples(int trainingSamples);

    int getTestingSamples() const;

    void setTestingSamples(int testingSamples);
};


#endif //PSGDC_DATASET_H
