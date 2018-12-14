//
// Created by vibhatha on 11/5/18.
//

#ifndef PSGDC_UTIL_H
#define PSGDC_UTIL_H

#include <iostream>

using namespace std;

class Util {
public:
    Util();
    void print2DMatrix(double** x, int row, int column);
    void print1DMatrix(double* x, int features);
    void writeWeight(double* w, int size, string file);
    void writeWeightEpochLog(double* w, int size, string file);
    void writeAccuracyPerEpoch(double epoch, double acc, string file);
    void writeLossAccuracyPerEpoch(double epoch, double acc, double cost, string file);
    void writeTimeLossAccuracyPerEpoch(double epoch, double acc, double cost, double time, string file);
    void summary(string logfile, int world_size, double acc, double time, string datasource);
    void summary(string logfile, int world_size, double acc, double time, double alpha);
    void summary(string logfile, int world_size, double acc, double time);
    void compareChange(double* w_new, double* w_old, double* w_res, int features);
    void copyArray(double * a, double *b, int size);
    string getTimestamp();
    int seed();

};


#endif //PSGDC_UTIL_H
