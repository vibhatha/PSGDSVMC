//
// Created by vibhatha on 11/5/18.
//

#ifndef PSGDC_UTIL_H
#define PSGDC_UTIL_H

#include <iostream>
#include <vector>

using namespace std;

class Util {
public:
    Util();
    void print2DMatrix(double** x, int row, int column);
    void print1DMatrix(double* x, int features);
    void writeWeight(double* w, int size, string file);
    void writeAccuracyPerEpoch(double epoch, double acc, string file);
    void writeAccuracyPerEpoch(double epoch, double acc, double dot_prod_time, double weight_update_time, double cost_calculate_time, double convergence_calculate_time, double predict_time, string file);
    void writeLossAccuracyPerEpoch(double epoch, double acc, double cost, string file);
    void writeTimeLossAccuracyPerEpoch(double epoch, double acc, double cost, double time, string file);
    void summary(string logfile, int world_size, double acc, double time, string datasource);
    void summary(string logfile, int world_size, double acc, double time, double alpha, double error, int effective_epochs);
    void summary(string logfile, int world_size, double acc, double time, double alpha, double error);
    void summary(string logfile, int world_size, double acc, double time, double alpha);
    void summary(string logfile, int world_size, double acc, double time);
    string getTimestamp();
    void averageWeight(vector<double*> weights, int features, double* w);
    void copyArray(double* source, double* copy, int features);
    int seed();

};


#endif //PSGDC_UTIL_H
