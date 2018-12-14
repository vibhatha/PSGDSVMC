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
    cout << "* ";
    for (int i = 0; i < features; ++i) {
        cout  << x[i] << " ";
    }
    cout << "* ";
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

void Util::writeWeightEpochLog(double *w, int size, string logfile) {
    string timestamp = getTimestamp();
    string line;
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

void Util::writeAccuracyPerEpoch(double epoch, double acc, string file) {
    ofstream myfile(file, ios::out | ios::app);
    if (myfile.is_open()) {
        myfile << epoch <<","<<acc<< "\n";
        myfile.close();
    }
}

void Util::writeLossAccuracyPerEpoch(double epoch, double acc, double cost, string file) {
    ofstream myfile(file, ios::out | ios::app);
    if (myfile.is_open()) {
        myfile << epoch <<","<<acc<<","<<cost<<"\n";
        myfile.close();
    }
}

void Util::writeTimeLossAccuracyPerEpoch(double epoch, double acc, double cost, double time, string file) {
    ofstream myfile(file, ios::out | ios::app);
    if (myfile.is_open()) {
        myfile << epoch <<","<<acc<<","<<cost<<","<<time<<","<<"\n";
        myfile.close();
    }
}

int Util::seed() {
    static int i = 1;
    return i++;
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


void Util::summary(string logfile, int world_size, double acc, double time) {
    ofstream myfile(logfile, ios::out | ios::app);
    string timestamp = getTimestamp();
    if (myfile.is_open()) {

        myfile << world_size << "," << time << "," << acc << "," << timestamp << "\n";

        myfile.close();
    }
}

void Util::summary(string logfile, int world_size, double acc, double time, double alpha) {
    ofstream myfile(logfile, ios::out | ios::app);
    string timestamp = getTimestamp();
    if (myfile.is_open()) {

        myfile << world_size << "," << time << "," << acc << "," << alpha << "," << timestamp << "\n";

        myfile.close();
    }
}

void Util::compareChange(double *w_new, double *w_old, double *w_res, int features) {
    for (int i = 0; i < features; ++i) {
        w_res[i] = w_new[i] - w_old[i];
    }
}

void Util::copyArray(double *a, double *b, int size) {
    for (int i = 0; i < size; ++i) {
        b[i] = a[i];
    }
}