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

const string &ResourceManager::getLogSourceBasePath() const {
    return logSourceBasePath;
}

void ResourceManager::setLogSourceBasePath(const string &logSourceBasePath) {
    ResourceManager::logSourceBasePath = logSourceBasePath;
}
