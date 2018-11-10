//
// Created by vibhatha on 11/5/18.
//

#include <iostream>
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