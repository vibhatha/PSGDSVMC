//
// Created by vibhatha on 11/5/18.
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "DataSet.h"

using namespace std;

DataSet::DataSet(string sourceFile_, int features_, int trainingSamples_, int testingSamples_) {
    sourceFile = sourceFile_;
    features = features_;
    trainingSamples = trainingSamples_;
    testingSamples = testingSamples_;
}

DataSet::DataSet(string sourceFile_, int features_, int trainingSamples_, int testingSamples_,
                 bool isSplit_, double ratio_) {
    sourceFile = sourceFile_;
    features = features_;
    trainingSamples = trainingSamples_;
    testingSamples = testingSamples_;
    isSplit = isSplit_;
    ratio = ratio_;
}

DataSet::DataSet(string sourceFile_, int features_, int trainingSamples_, bool isSplit_,
                 double ratio_) {
    sourceFile = sourceFile_;
    features = features_;
    trainingSamples = trainingSamples_;
    isSplit = isSplit_;
    ratio = ratio_;
}

void DataSet::load() {

    Xtrain = new double*[trainingSamples];
    ytrain = new double[trainingSamples];
    ifstream file(sourceFile);
    cout << "Loading File : " << sourceFile << endl;
    for(int row = 0; row < trainingSamples; row++)
    {
        Xtrain[row] = new double[features];
        string line;
        getline(file, line);
        if ( !file.good() ){
            printf("File is not readable \n");
            break;
        }

        //cout << line << endl;
        vector<double> vect;

        std::stringstream ss(line);

        double i;

        while (ss >> i)
        {
            vect.push_back(i);

            if (ss.peek() == ',')
                ss.ignore();
        }
        ytrain[row] = vect.at(0);

        for (int j=1; j< vect.size(); j++){
            Xtrain[row][j-1] = vect.at(j);
        }
    }
}

double** DataSet::getXtest() {
    return Xtest;
}

double** DataSet::getXtrain() {
    return Xtrain;
}

double* DataSet::getYtest() {
    return ytest;
}

double* DataSet::getYtrain() {
    return ytrain;
}