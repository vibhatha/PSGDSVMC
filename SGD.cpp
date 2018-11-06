//
// Created by vibhatha on 11/5/18.
//

#include <iostream>
#include "SGD.h"

using namespace std;

SGD::SGD(double **Xn, double* yn, double alphan, int itrN) {
    X = Xn;
    y=yn;
    alpha = alphan;
    iterations = itrN;
}

void SGD::sgd() {
    printf("Iterations %d ", iterations);
}
