//
// Created by vibhatha on 11/11/18.
//

#include "ResourceManager.h"
#include <fstream>
#include <string>
#include <iostream>

using namespace std;

ResourceManager::ResourceManager() {}

const string &ResourceManager::getDataSourceBasePath() const {
    return dataSourceBasePath;
}

void ResourceManager::setDataSourceBasePath(const string &dataSourceBasePath) {
    ResourceManager::dataSourceBasePath = dataSourceBasePath;
}

void ResourceManager::loadDataSourcePath() {
    string resourceFile = "datasource.yaml";
    std::ifstream file(resourceFile);
    std::string str;
    string line;
    while (std::getline(file, str))
    {
        line = str;
    }

    string delimiter = ": ";
    std::string token = line.substr(line.find(delimiter) + delimiter.length(),line.length());
    this->setDataSourceBasePath(token);

}

void ResourceManager::loadLogSourcePath() {
    string resourceFile = "logsource.yaml";
    std::ifstream file(resourceFile);
    std::string str;
    string line;
    while (std::getline(file, str))
    {
        line = str;
    }

    string delimiter = ": ";
    std::string token = line.substr(line.find(delimiter) + delimiter.length(),line.length());
    this->setLogSourceBasePath(token);
}

void ResourceManager::loadSummaryPath() {
    string resourceFile = "logsummary.yaml";
    std::ifstream file(resourceFile);
    std::string str;
    string line;
    while (std::getline(file, str))
    {
        line = str;
    }

    string delimiter = ": ";
    std::string token = line.substr(line.find(delimiter) + delimiter.length(),line.length());
    this->setLogSummaryBasePath(token);
}

void ResourceManager::loadWeightSummaryPath() {
    string resourceFile = "weightsummary.yaml";
    std::ifstream file(resourceFile);
    std::string str;
    string line;
    while (std::getline(file, str))
    {
        line = str;
    }

    string delimiter = ": ";
    std::string token = line.substr(line.find(delimiter) + delimiter.length(),line.length());
    this->setWeightSummaryBasePath(token);
}

void ResourceManager::loadEpochSummaryPath() {
    string resourceFile = "epochlogsummary.yaml";
    std::ifstream file(resourceFile);
    std::string str;
    string line;
    while (std::getline(file, str))
    {
        line = str;
    }

    string delimiter = ": ";
    std::string token = line.substr(line.find(delimiter) + delimiter.length(),line.length());
    this->setEpochlogSummaryBasePath(token);
}

const string &ResourceManager::getLogSourceBasePath() const {
    return logSourceBasePath;
}

void ResourceManager::setLogSourceBasePath(const string &logSourceBasePath) {
    ResourceManager::logSourceBasePath = logSourceBasePath;
}

const string &ResourceManager::getLogSummaryBasePath() const {
    return logSummaryBasePath;
}

void ResourceManager::setLogSummaryBasePath(const string &logSummaryBasePath) {
    ResourceManager::logSummaryBasePath = logSummaryBasePath;
}

const string &ResourceManager::getWeightSummaryBasePath() const {
    return weightSummaryBasePath;
}

void ResourceManager::setWeightSummaryBasePath(const string &weightSummaryBasePath) {
    ResourceManager::weightSummaryBasePath = weightSummaryBasePath;
}

const string &ResourceManager::getEpochlogSummaryBasePath() const {
    return epochlogSummaryBasePath;
}

void ResourceManager::setEpochlogSummaryBasePath(const string &epochlogSummaryBasePath) {
    ResourceManager::epochlogSummaryBasePath = epochlogSummaryBasePath;
}
