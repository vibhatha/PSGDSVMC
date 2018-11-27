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
    string logSummaryBasePath;
    string weightSummaryBasePath;
    string epochlogSummaryBasePath;

public:
    ResourceManager();

    void loadDataSourcePath();

    void loadLogSourcePath();

    void loadSummaryPath();

    void loadWeightSummaryPath();

    void loadEpochSummaryPath();

    void setDataSourceBasePath(const string &dataSourceBasePath);

    const string &getDataSourceBasePath() const;

    const string &getLogSourceBasePath() const;

    void setLogSourceBasePath(const string &logSourceBasePath);

    const string &getLogSummaryBasePath() const;

    void setLogSummaryBasePath(const string &logSummaryBasePath);

    const string &getWeightSummaryBasePath() const;

    void setWeightSummaryBasePath(const string &weightSummaryBasePath);

    const string &getEpochlogSummaryBasePath() const;

    void setEpochlogSummaryBasePath(const string &epochlogSummaryBasePath);

};


#endif //PSGDC_RESOURCEMANAGER_H
