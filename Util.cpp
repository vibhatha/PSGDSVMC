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

void Util::summary(string logfile, int world_size, double acc, double time, double alpha, double error) {
    ofstream myfile(logfile, ios::out | ios::app);
    string timestamp = getTimestamp();
    if (myfile.is_open()) {

        myfile << world_size << "," << time << "," << acc << "," << alpha << "," << "," << error << "," << timestamp << "\n";

        myfile.close();
    }
}

void Util::averageWeight(vector<double *> weights, int features, double *w) {
        int block_size = weights.size();
        //cout << "Size : " << block_size << " ";
        for (int j = 0; j < features; ++j) {
            for (int i = 0; i < block_size; ++i) {
                w[j] = weights.at(i)[j];
            }
            w[j]/=block_size;
            //cout << w[j] << " ";
        }
    cout << "------------------------" << endl;
    for (int k = 0; k < block_size; ++k) {
        for (int i = 0; i < features; ++i) {
            cout << weights.at(k)[i] << " ";
        }
        cout << endl;
    }
    cout << "------------------------" << endl;


}

void Util::copyArray(double *source, double *copy, int features) {
    for (int i = 0; i < features; ++i) {
        copy[i] = source[i];
    }
}