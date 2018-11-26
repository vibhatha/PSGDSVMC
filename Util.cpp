//
// Created by vibhatha on 11/5/18.
//

#include <iostream>
#include <fstream>
#include "Util.h"

using namespace std;

Util::Util() {

}

void Util::print2DMatrix(double** x, int row, int column) {
    for (int i = 0; i < row; ++i) {
        cout << i << " : ";
        for (int j = 0; j < column; ++j) {
            cout << x[i][j] << " ";
        }
        cout << "\n";
    }
    cout << "\n";
}

void Util::print1DMatrix(double* x, int features) {
    for (int i = 0; i < features; ++i) {
        cout  << x[i] << " ";
    }
    cout << "\n";
}

void Util::writeWeight(double *w, int size, string logfile) {
    string timestamp = getTimestamp();
    string line;
    logfile.append("_").append(timestamp);
    ofstream myfile(logfile, ios::out | ios::app);
    for (int i = 0; i < size; ++i) {
        if(i<size-2){
            line.append(to_string(w[i])).append(",");
        }else{
            line.append(to_string(w[i]));
        }
    }
    if (myfile.is_open()) {
        myfile << line << "\n";
        myfile.close();
    }
}

string Util::getTimestamp() {
    string string1;
    time_t t = time(0);   // get time now
    tm *now = localtime(&t);
    string datestring;
    datestring.append(to_string(now->tm_year + 1900)).append("-").append(to_string((now->tm_mon + 1))).append(
            "-").append(to_string(now->tm_mday));
    string timestring;
    timestring.append(to_string(now->tm_hour)).append(":").append(to_string(now->tm_min)).append(":").append(
            to_string(now->tm_sec));
    string1.append(datestring).append("__").append(timestring);
    return string1;
}
