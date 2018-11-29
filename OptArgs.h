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
    bool isEpochTime = false;
    bool isNormalTime = false;
    bool bulk = false;
    bool batch = false;
    double batch_per = 0.10;
    double drop_out_per = 0.25;
    bool isDrop = false;
    bool sequential = false;
    bool ring = false;
    bool randomringv1 = false;
    bool randomringv2 = false;
    bool fullbatchv1 = false;
    bool pegasos = false;


public:

    OptArgs();

    OptArgs(const string &dataset, int features, int trainingSamples, int testingSamples, double alpha, bool isSplit,
            double ratio, int threads, int workers);

    OptArgs(const string &dataset, int features, int trainingSamples, int testingSamples, double alpha, bool isSplit,
            double ratio, int threads, int workers, int iterations);

    OptArgs(const string &dataset, int features, int trainingSamples, int testingSamples, double alpha, bool isSplit,
            double ratio, int threads, int workers, int iterations, bool isEpochTime, bool isNormalTime);

    OptArgs(const string &dataset, int features, int trainingSamples, int testingSamples, double alpha, bool isSplit,
            double ratio, int threads, int workers, int iterations, bool isEpochTime, bool isNormalTime, bool bulk);

    OptArgs(const string &dataset, int features, int trainingSamples, int testingSamples, double alpha, bool isSplit,
            double ratio, int threads, int workers, int iterations, bool isEpochTime, bool isNormalTime, bool bulk,
            bool batch, double batch_per);

    OptArgs(const string &dataset, int features, int trainingSamples, int testingSamples, double alpha, bool isSplit,
            double ratio, int threads, int workers, int iterations, bool isEpochTime, bool isNormalTime, bool bulk,
            bool batch, double batch_per, double drop_out_per, bool isDrop);

    OptArgs(const string &dataset, int features, int trainingSamples, int testingSamples, double alpha, bool isSplit,
            double ratio, int threads, int workers, int iterations, bool isEpochTime, bool isNormalTime, bool bulk,
            bool batch, double batch_per, double drop_out_per, bool isDrop, bool sequential);

    OptArgs(const string &dataset, int features, int trainingSamples, int testingSamples, double alpha, bool isSplit,
            double ratio, int threads, int workers, int iterations, bool isEpochTime, bool isNormalTime, bool bulk,
            bool batch, double batch_per, double drop_out_per, bool isDrop, bool sequential, bool ring,
            bool randomringv1, bool randomringv2);

    OptArgs(const string &dataset, int features, int trainingSamples, int testingSamples, double alpha, bool isSplit,
            double ratio, int threads, int workers, int iterations, bool isEpochTime, bool isNormalTime, bool bulk,
            bool batch, double batch_per, double drop_out_per, bool isDrop, bool sequential, bool ring,
            bool randomringv1, bool randomringv2, bool fullbatchv1);

    OptArgs(const string &dataset, int features, int trainingSamples, int testingSamples, double alpha, bool isSplit,
            double ratio, int threads, int workers, int iterations, bool isEpochTime, bool isNormalTime, bool bulk,
            bool batch, double batch_per, double drop_out_per, bool isDrop, bool sequential, bool ring,
            bool randomringv1, bool randomringv2, bool fullbatchv1, bool pegasos);


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

    bool isIsEpochTime() const;

    void setIsEpochTime(bool isEpochTime);

    bool isIsNormalTime() const;

    void setIsNormalTime(bool isNormalTime);

    bool isBulk() const;

    void setBulk(bool bulk);

    bool isBatch() const;

    void setBatch(bool batch);

    double getBatch_per() const;

    void setBatch_per(double batch_per);

    double getDrop_out_per() const;

    void setDrop_out_per(double drop_out_per);

    bool isIsDrop() const;

    void setIsDrop(bool isDrop);

    bool isSequential() const;

    void setSequential(bool sequential);

    bool isRing() const;

    void setRing(bool ring);

    bool isRandomringv1() const;

    void setRandomringv1(bool randomringv1);

    bool isRandomringv2() const;

    void setRandomringv2(bool randomringv2);

    bool isFullbatchv1() const;

    void setFullbatchv1(bool fullbatchv1);

    bool isPegasos() const;

    void setPegasos(bool pegasos);


};


#endif //PSGDC_OPTARGS_H
