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

    if(isSplit== false){
        Xtrain = new double*[trainingSamples];
        ytrain = new double[trainingSamples];
        ifstream file(trainFile);
        cout << "Loading File : " << trainFile << endl;
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

        Xtest = new double*[testingSamples];
        ytest = new double[testingSamples];
        ifstream fileTest(testFile);
        cout << "Loading File : " << testFile << endl;
        for(int row = 0; row < testingSamples; row++)
        {
            Xtest[row] = new double[features];
            string line;
            getline(fileTest, line);
            if ( !fileTest.good() ){
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
            ytest[row] = vect.at(0);

            for (int j=1; j< vect.size(); j++){
                Xtest[row][j-1] = vect.at(j);
            }
        }
    }

    if(isSplit == true) {
        printf("Splitting data ... \n");
        int totalSamples = trainingSamples;
        int trainSet = totalSamples * ratio;
        int testSet = totalSamples - trainSet;
        printf("Train and Test %d, %d \n", trainSet, testSet);
        this->setTestingSamples(testSet);
        this->setTrainingSamples(trainSet);
        Xtrain = new double*[trainSet];
        ytrain = new double[trainSet];
        Xtest = new double*[testSet];
        ytest = new double[testSet];
        ifstream file(sourceFile);
        cout << "Loading File : " << sourceFile << endl;
        int rowTest = 0;
        for(int row = 0; row < totalSamples; row++)
        {
            if(row<trainSet){
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

            if(row>=trainSet and rowTest <= testSet){
                cout << "Row : " << row << ", Row Test Id  : " << rowTest << endl;
                Xtest[rowTest] = new double[features];
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
                ytest[rowTest] = vect.at(0);

                for (int j=1; j< vect.size(); j++){
                    Xtest[rowTest][j-1] = vect.at(j);
                }
                rowTest++;
            }

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

DataSet::DataSet(int features, int trainingSamples, int testingSamples, const string &trainFile, const string &testFile)
        : features(features), trainingSamples(trainingSamples), testingSamples(testingSamples), trainFile(trainFile),
          testFile(testFile) {}

int DataSet::getTrainingSamples() const {
    return trainingSamples;
}

void DataSet::setTrainingSamples(int trainingSamples) {
    DataSet::trainingSamples = trainingSamples;
}

int DataSet::getTestingSamples() const {
    return testingSamples;
}

void DataSet::setTestingSamples(int testingSamples) {
    DataSet::testingSamples = testingSamples;
}

