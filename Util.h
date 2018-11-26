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
    string getTimestamp();

};


#endif //PSGDC_UTIL_H
