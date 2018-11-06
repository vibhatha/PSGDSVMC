//
// Created by vibhatha on 11/5/18.
//

#include <iostream>
#include "Util.h"

using namespace std;

Util::Util() {

}

void Util::printX(double** x, int column, int row) {


    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            printf("%f ",x[i][j]);
        }
        printf("\n");
    }



}