//
// Created by vibhatha on 11/7/18.
//

#ifndef PSGDC_OPTARGS_H
#define PSGDC_OPTARGS_H

#include <iostream>

using namespace std;

class OptArgs {
private:
    string dataset;
    int features;
    int trainingSamples;
    int testingSamples;
    double alpha;
    bool isSplit = false;
    double ratio = 0.80;
    int threads = 1;
    int workers = 1;
    int iterations;

public:

    OptArgs();

    OptArgs(const string &dataset, int features, int trainingSamples, int testingSamples, double alpha, bool isSplit,
            double ratio, int threads, int workers);

    OptArgs(const string &dataset, int features, int trainingSamples, int testingSamples, double alpha, bool isSplit,
            double ratio, int threads, int workers, int iterations);

    const string &getDataset() const;

    void setDataset(const string &dataset);

    int getFeatures() const;

    void setFeatures(int features);

    int getTrainingSamples() const;

    void setTrainingSamples(int trainingSamples);

    int getTestingSamples() const;

    void setTestingSamples(int testingSamples);

    double getAlpha() const;

    void setAlpha(double alpha);

    bool isIsSplit() const;

    void setIsSplit(bool isSplit);

    double getRatio() const;

    void setRatio(double ratio);

    int getThreads() const;

    void setThreads(int threads);

    int getWorkers() const;

    void setWorkers(int workers);

    int getIterations() const;

    void setIterations(int iterations);

    void toString();


};


#endif //PSGDC_OPTARGS_H
