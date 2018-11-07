//
// Created by vibhatha on 11/7/18.
//

#ifndef PSGDC_ARGREADER_H
#define PSGDC_ARGREADER_H


#include "OptArgs.h"

class ArgReader {
private:
    int argc;
    char** argv;
public:

    ArgReader(int argc, char **argv);
    OptArgs getParams();

};


#endif //PSGDC_ARGREADER_H
