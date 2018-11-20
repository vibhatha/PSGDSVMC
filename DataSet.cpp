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

void DataSet::distributedLoad() {
    if (isSplit == false) {

    }

    if (isSplit == true) {
        printf("Splitting data ... \n");
        int totalSamples = trainingSamples;
        int trainingSet = trainingSamples * ratio;
        int testingSet = totalSamples - trainingSet;
        int dataPerMachine = trainingSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;
        int start = world_rank * dataPerMachine;
        int end = start + dataPerMachine;
        this->setDataPerMachine(dataPerMachine);
        if(world_rank==0) {
            cout << "Loading Data in Rank " << world_rank << ", Start :  " << start << ", End " << end
                 << ",Data per Rank : " << dataPerMachine << endl;
            cout << "Loading File : " << trainFile << endl;
        }

        this->setTestingSamples(testingSet);
        this->setTrainingSamples(dataPerMachine);
        Xtrain = new double *[dataPerMachine];
        ytrain = new double[dataPerMachine];


        ifstream file(trainFile);

        int rowTest = 0;
        for (int row = 0; row < totalVisibleSamples; row++) {


            string line;
            getline(file, line);
            if (!file.good()) {
                printf("File is not readable \n");
                break;
            }

            if (row >= start and row < end) {
                Xtrain[row - start] = new double[features];
                vector<double> vect;

                std::stringstream ss(line);

                double i;

                while (ss >> i) {
                    vect.push_back(i);

                    if (ss.peek() == ',')
                        ss.ignore();
                }
                ytrain[row - start] = vect.at(0);

                for (int j = 1; j < vect.size(); j++) {
                    Xtrain[row - start][j - 1] = vect.at(j);
                }
            }

        }
    }
}

void DataSet::loadTestData(double **Xtest, double *ytest) {
    ifstream file2(testFile);
    if(world_rank==0) {
        cout << "Loading File : " << testFile << endl;
    }


    for (int row = 0; row < testingSamples; row++) {
        //cout << "Rank : " << world_rank << ", row : " << row << endl;
        string line;
        getline(file2, line);
        if (!file2.good()) {
            printf("File is not readable \n");
            break;
        }

        //cout << "start : " << (row-start) << " End : " << end << endl;
        vector<double> vect;

        std::stringstream ss(line);

        double i;

        while (ss >> i) {
            vect.push_back(i);

            if (ss.peek() == ',')
                ss.ignore();
        }
        ytest[row] = vect.at(0);

        for (int j = 1; j < vect.size(); j++) {
            Xtest[row][j - 1] = vect.at(j);
        }


    }
}


void DataSet::distributedLoad(double **Xtrain, double *ytrain, double **Xtest, double *ytest) {
    if (isSplit == false) {
        int totalSamples = trainingSamples;
        int trainingSet = trainingSamples;
        int testingSet = testingSamples;
        int dataPerMachine = trainingSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;
        int start = world_rank * dataPerMachine;
        int end = start + dataPerMachine;
        this->setDataPerMachine(dataPerMachine);
        cout << "Loading Data in Rank " << world_rank << ", Start :  " << start << ", End " << end << ", M : " << dataPerMachine << endl;
        this->setTestingSamples(testingSet);
        this->setTrainingSamples(dataPerMachine);

        ifstream file(trainFile);
        cout << "Loading File : " << trainFile << endl;
        int rowTest = 0;
        for (int row = 0; row < totalVisibleSamples; row++) {
            //cout << "Rank : " << world_rank << ", row : " << row << endl;
            string line;
            getline(file, line);
            if (!file.good()) {
                printf("File is not readable \n");
                break;
            }

            if (row >= start and row < end and row < totalVisibleSamples) {
                //cout << "start : " << (row-start) << " End : " << end << endl;
                vector<double> vect;

                std::stringstream ss(line);

                double i;

                while (ss >> i) {
                    vect.push_back(i);

                    if (ss.peek() == ',')
                        ss.ignore();
                }
                ytrain[row - start] = vect.at(0);

                for (int j = 1; j < vect.size(); j++) {
                    Xtrain[row - start][j - 1] = vect.at(j);
                }
            }

        }
        if(this->isBulk==true) {
            cout << "Bulk Files are being used ..." << endl;
        }
        if(this->isBulk==false) {
            ifstream file2(testFile);
            cout << "Loading File : " << testFile << endl;

            for (int row = 0; row < testingSamples; row++) {
                //cout << "Rank : " << world_rank << ", row : " << row << endl;
                string line;
                getline(file2, line);
                if (!file2.good()) {
                    printf("File is not readable \n");
                    break;
                }

                //cout << "start : " << (row-start) << " End : " << end << endl;
                vector<double> vect;

                std::stringstream ss(line);

                double i;

                while (ss >> i) {
                    vect.push_back(i);

                    if (ss.peek() == ',')
                        ss.ignore();
                }
                ytest[row] = vect.at(0);

                for (int j = 1; j < vect.size(); j++) {
                    Xtest[row][j - 1] = vect.at(j);
                }


            }
        }

    }

    if (isSplit == true) {
        printf("Splitting data ... \n");
        int totalSamples = trainingSamples;
        int trainingSet = trainingSamples * ratio;
        int testingSet = totalSamples - trainingSet;
        int dataPerMachine = trainingSet / world_size;
        int totalVisibleSamples = dataPerMachine * world_size;
        int start = world_rank * dataPerMachine;
        int end = start + dataPerMachine;
        this->setDataPerMachine(dataPerMachine);

        cout << "Loading Data in Rank " << world_rank << ", Start :  " << start << ", End " << end << endl;
        this->setTestingSamples(testingSet);
        this->setTrainingSamples(dataPerMachine);

        ifstream file(trainFile);
        cout << "Loading File : " << trainFile << endl;
        int rowTest = 0;
        for (int row = 0; row < totalSamples; row++) {
            //cout << "Rank : " << world_rank << ", row : " << row << endl;
            string line;
            getline(file, line);
            if (!file.good()) {
                printf("File is not readable \n");
                break;
            }

            if (row >= start and row < end and row < totalVisibleSamples) {
                //cout << "start : " << (row-start) << " End : " << end << endl;
                vector<double> vect;

                std::stringstream ss(line);

                double i;

                while (ss >> i) {
                    vect.push_back(i);

                    if (ss.peek() == ',')
                        ss.ignore();
                }
                ytrain[row - start] = vect.at(0);

                for (int j = 1; j < vect.size(); j++) {
                    Xtrain[row - start][j - 1] = vect.at(j);
                }
            } else if (row > trainingSet) {
                //cout << line << endl;
                vector<double> vect;

                std::stringstream ss(line);

                double i;

                while (ss >> i) {
                    vect.push_back(i);

                    if (ss.peek() == ',')
                        ss.ignore();
                }
                ytest[row - trainingSet] = vect.at(0);

                for (int j = 1; j < vect.size(); j++) {
                    Xtest[row - trainingSet][j - 1] = vect.at(j);
                }
            }

        }


    }
}


void DataSet::load() {

    if (isSplit == false) {
        Xtrain = new double *[trainingSamples];
        ytrain = new double[trainingSamples];
        ifstream file(trainFile);
        cout << "Loading File : " << trainFile << endl;
        for (int row = 0; row < trainingSamples; row++) {
            Xtrain[row] = new double[features];
            string line;
            getline(file, line);
            if (!file.good()) {
                printf("File is not readable \n");
                break;
            }

            //cout << line << endl;
            vector<double> vect;

            std::stringstream ss(line);

            double i;

            while (ss >> i) {
                vect.push_back(i);

                if (ss.peek() == ',')
                    ss.ignore();
            }
            ytrain[row] = vect.at(0);

            for (int j = 1; j < vect.size(); j++) {
                Xtrain[row][j - 1] = vect.at(j);
            }
        }

        Xtest = new double *[testingSamples];
        ytest = new double[testingSamples];
        ifstream fileTest(testFile);
        cout << "Loading File : " << testFile << endl;
        for (int row = 0; row < testingSamples; row++) {
            Xtest[row] = new double[features];
            string line;
            getline(fileTest, line);
            if (!fileTest.good()) {
                printf("File is not readable \n");
                break;
            }

            //cout << line << endl;
            vector<double> vect;

            std::stringstream ss(line);

            double i;

            while (ss >> i) {
                vect.push_back(i);

                if (ss.peek() == ',')
                    ss.ignore();
            }
            ytest[row] = vect.at(0);

            for (int j = 1; j < vect.size(); j++) {
                Xtest[row][j - 1] = vect.at(j);
            }
        }
    }

    if (isSplit == true) {
        printf("Splitting data ... \n");
        int totalSamples = trainingSamples;
        int trainSet = totalSamples * ratio;
        int testSet = totalSamples - trainSet;
        printf("Train and Test %d, %d \n", trainSet, testSet);
        this->setTestingSamples(testSet);
        this->setTrainingSamples(trainSet);
        Xtrain = new double *[trainSet];
        ytrain = new double[trainSet];
        Xtest = new double *[testSet];
        ytest = new double[testSet];
        ifstream file(sourceFile);
        cout << "Loading File : " << sourceFile << endl;
        int rowTest = 0;
        for (int row = 0; row < totalSamples; row++) {
            if (row < trainSet) {
                Xtrain[row] = new double[features];
                string line;
                getline(file, line);
                if (!file.good()) {
                    printf("File is not readable \n");
                    break;
                }

                //cout << line << endl;
                vector<double> vect;

                std::stringstream ss(line);

                double i;

                while (ss >> i) {
                    vect.push_back(i);

                    if (ss.peek() == ',')
                        ss.ignore();
                }
                ytrain[row] = vect.at(0);

                for (int j = 1; j < vect.size(); j++) {
                    Xtrain[row][j - 1] = vect.at(j);
                }
            }

            if (row >= trainSet and rowTest <= testSet) {
                //cout << "Row : " << row << ", Row Test Id  : " << rowTest << endl;
                Xtest[rowTest] = new double[features];
                string line;
                getline(file, line);
                if (!file.good()) {
                    printf("File is not readable \n");
                    break;
                }

                //cout << line << endl;
                vector<double> vect;

                std::stringstream ss(line);

                double i;

                while (ss >> i) {
                    vect.push_back(i);

                    if (ss.peek() == ',')
                        ss.ignore();
                }
                ytest[rowTest] = vect.at(0);

                for (int j = 1; j < vect.size(); j++) {
                    Xtest[rowTest][j - 1] = vect.at(j);
                }
                rowTest++;
            }

        }
    }


}


void DataSet::load(double **Xtrain, double *ytrain, double **Xtest, double *ytest) {

    if (isSplit == false) {
        ifstream file(trainFile);
        cout << "Loading File : " << trainFile << endl;
        for (int row = 0; row < trainingSamples; row++) {

            string line;
            getline(file, line);
            if (!file.good()) {
                printf("File is not readable \n");
                break;
            }

            //cout << line << endl;
            vector<double> vect;

            std::stringstream ss(line);

            double i;

            while (ss >> i) {
                vect.push_back(i);

                if (ss.peek() == ',')
                    ss.ignore();
            }
            ytrain[row] = vect.at(0);

            for (int j = 1; j < vect.size(); j++) {
                Xtrain[row][j - 1] = vect.at(j);
            }
        }


        ifstream fileTest(testFile);
        cout << "Loading File : " << testFile << endl;
        for (int row = 0; row < testingSamples; row++) {

            string line;
            getline(fileTest, line);
            if (!fileTest.good()) {
                printf("File is not readable \n");
                break;
            }

            //cout << line << endl;
            vector<double> vect;

            std::stringstream ss(line);

            double i;

            while (ss >> i) {
                vect.push_back(i);

                if (ss.peek() == ',')
                    ss.ignore();
            }
            ytest[row] = vect.at(0);

            for (int j = 1; j < vect.size(); j++) {
                Xtest[row][j - 1] = vect.at(j);
            }
        }
    }

    if (isSplit == true) {
        printf("Splitting data ... \n");
        int totalSamples = trainingSamples;
        int trainSet = totalSamples * ratio;
        int testSet = totalSamples - trainSet;
        printf("Train and Test %d, %d \n", trainSet, testSet);
        this->setTestingSamples(testSet);
        this->setTrainingSamples(trainSet);

        ifstream file(sourceFile);
        cout << "Loading File : " << sourceFile << endl;
        int rowTest = 0;
        for (int row = 0; row < totalSamples; row++) {
            if (row < trainSet) {

                string line;
                getline(file, line);
                if (!file.good()) {
                    printf("File is not readable \n");
                    break;
                }

                //cout << line << endl;
                vector<double> vect;

                std::stringstream ss(line);

                double i;

                while (ss >> i) {
                    vect.push_back(i);

                    if (ss.peek() == ',')
                        ss.ignore();
                }
                ytrain[row] = vect.at(0);

                for (int j = 1; j < vect.size(); j++) {
                    Xtrain[row][j - 1] = vect.at(j);
                }
            }

            if (row >= trainSet and rowTest <= testSet) {
                //cout << "Row : " << row << ", Row Test Id  : " << rowTest << endl;

                string line;
                getline(file, line);
                if (!file.good()) {
                    printf("File is not readable \n");
                    break;
                }

                //cout << line << endl;
                vector<double> vect;

                std::stringstream ss(line);

                double i;

                while (ss >> i) {
                    vect.push_back(i);

                    if (ss.peek() == ',')
                        ss.ignore();
                }
                ytest[rowTest] = vect.at(0);

                for (int j = 1; j < vect.size(); j++) {
                    Xtest[rowTest][j - 1] = vect.at(j);
                }
                rowTest++;
            }

        }
    }


}

double **DataSet::getXtest() {
    return Xtest;
}

double **DataSet::getXtrain() {
    return Xtrain;
}

double *DataSet::getYtest() {
    return ytest;
}

double *DataSet::getYtrain() {
    return ytrain;
}

DataSet::DataSet(int features, int trainingSamples, int testingSamples, const string &trainFile, const string &testFile)
        : features(features), trainingSamples(trainingSamples), testingSamples(testingSamples), trainFile(trainFile),
          testFile(testFile), isSplit(false) {}

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

DataSet::DataSet(int features, bool isSplit, double ratio, const string &trainFile, const string &testFile,
                 int world_size, int world_rank) : features(features), isSplit(isSplit), ratio(ratio),
                                                   trainFile(trainFile), testFile(testFile), world_size(world_size),
                                                   world_rank(world_rank) {}

DataSet::DataSet(int features, int trainingSamples, int testingSamples, bool isSplit, double ratio,
                 const string &trainFile, int world_size, int world_rank) : features(features),
                                                                            trainingSamples(trainingSamples),
                                                                            testingSamples(testingSamples),
                                                                            isSplit(isSplit), ratio(ratio),
                                                                            trainFile(trainFile),
                                                                            world_size(world_size),
                                                                            world_rank(world_rank) {}

int DataSet::getDataPerMachine() const {
    return dataPerMachine;
}

void DataSet::setDataPerMachine(int dataPerMachine) {
    DataSet::dataPerMachine = dataPerMachine;
}

DataSet::DataSet(int features, int trainingSamples, int testingSamples, const string &trainFile, const string &testFile,
                 int world_size, int world_rank) : features(features), trainingSamples(trainingSamples),
                                                   testingSamples(testingSamples), trainFile(trainFile),
                                                   testFile(testFile), world_size(world_size), world_rank(world_rank) {}

DataSet::DataSet(int features, int testingSamples, const string &testFile) : features(features),
                                                                             testingSamples(testingSamples),
                                                                             testFile(testFile) {}

DataSet::DataSet(int features, int trainingSamples, int testingSamples, const string &trainFile, const string &testFile,
                 int world_size, int world_rank, bool isBulk) : features(features), trainingSamples(trainingSamples),
                                                                testingSamples(testingSamples), trainFile(trainFile),
                                                                testFile(testFile), world_size(world_size),
                                                                world_rank(world_rank), isBulk(isBulk) {}

