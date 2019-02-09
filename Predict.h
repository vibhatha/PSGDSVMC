//
// Created by vibhatha on 11/10/18.
//

#ifndef PSGDC_PREDICT_H
#define PSGDC_PREDICT_H


class Predict {

private:
    double** X;
    double* y;
    double accuracy;
    double* w;
    int testingSamples;
    int features;

public:
    Predict(double **X, double *y, double *w, int testingSamples, int features);

    double predict();
    double crossValidate();
    double testPrediction();


};


#endif //PSGDC_PREDICT_H
