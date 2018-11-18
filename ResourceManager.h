//
// Created by vibhatha on 11/11/18.
//

#ifndef PSGDC_RESOURCEMANAGER_H
#define PSGDC_RESOURCEMANAGER_H

#include <iostream>

using namespace std;


class ResourceManager {
private:
    string dataSourceBasePath;
    string logSourceBasePath;

public:
    ResourceManager();

    void loadDataSourcePath();

    void loadLogSourcePath();

    void setDataSourceBasePath(const string &dataSourceBasePath);

    const string &getDataSourceBasePath() const;

    const string &getLogSourceBasePath() const;

    void setLogSourceBasePath(const string &logSourceBasePath);

};


#endif //PSGDC_RESOURCEMANAGER_H
