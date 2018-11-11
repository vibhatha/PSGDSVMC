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

public:
    ResourceManager();

    void loadDataSourcePath();

    void setDataSourceBasePath(const string &dataSourceBasePath);

    const string &getDataSourceBasePath() const;

};


#endif //PSGDC_RESOURCEMANAGER_H
