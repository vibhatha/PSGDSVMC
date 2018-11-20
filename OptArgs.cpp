//
// Created by vibhatha on 11/7/18.
//

#include "OptArgs.h"

OptArgs::OptArgs() {

}

const string &OptArgs::getDataset() const {
    return dataset;
}

void OptArgs::setDataset(const string &dataset) {
    OptArgs::dataset = dataset;
}

int OptArgs::getFeatures() const {
    return features;
}

void OptArgs::setFeatures(int features) {
    OptArgs::features = features;
}

int OptArgs::getTrainingSamples() const {
    return trainingSamples;
}

void OptArgs::setTrainingSamples(int trainingSamples) {
    OptArgs::trainingSamples = trainingSamples;
}

int OptArgs::getTestingSamples() const {
    return testingSamples;
}

void OptArgs::setTestingSamples(int testingSamples) {
    OptArgs::testingSamples = testingSamples;
}

double OptArgs::getAlpha() const {
    return alpha;
}

void OptArgs::setAlpha(double alpha) {
    OptArgs::alpha = alpha;
}

bool OptArgs::isIsSplit() const {
    return isSplit;
}

void OptArgs::setIsSplit(bool isSplit) {
    OptArgs::isSplit = isSplit;
}

double OptArgs::getRatio() const {
    return ratio;
}

void OptArgs::setRatio(double ratio) {
    OptArgs::ratio = ratio;
}

int OptArgs::getThreads() const {
    return threads;
}

void OptArgs::setThreads(int threads) {
    OptArgs::threads = threads;
}

int OptArgs::getWorkers() const {
    return workers;
}

void OptArgs::setWorkers(int workers) {
    OptArgs::workers = workers;
}

int OptArgs::getIterations() const {
    return iterations;
}

void OptArgs::setIterations(int iterations) {
    OptArgs::iterations = iterations;
}

OptArgs::OptArgs(const string &dataset, int features, int trainingSamples, int testingSamples, double alpha,
                 bool isSplit, double ratio, int threads, int workers) : dataset(dataset), features(features),
                                                                         trainingSamples(trainingSamples),
                                                                         testingSamples(testingSamples), alpha(alpha),
                                                                         isSplit(isSplit), ratio(ratio),
                                                                         threads(threads), workers(workers) {}

OptArgs::OptArgs(const string &dataset, int features, int trainingSamples, int testingSamples, double alpha,
                 bool isSplit, double ratio, int threads, int workers, int iterations) : dataset(dataset),
                                                                                         features(features),
                                                                                         trainingSamples(
                                                                                                 trainingSamples),
                                                                                         testingSamples(testingSamples),
                                                                                         alpha(alpha), isSplit(isSplit),
                                                                                         ratio(ratio), threads(threads),
                                                                                         workers(workers),
                                                                                         iterations(iterations) {}
void OptArgs::toString() {
    cout << "Dataset : " << this->getDataset() << endl;
    cout << "Iterations : " << this->getIterations() << endl;
    cout << "Alpha : " << this->getAlpha() << endl;
    cout << "Features : " << this->getFeatures() << endl;
    cout << "Training Samples : " << this->getTrainingSamples() << endl;
    cout << "Testing Samples : " << this->getTestingSamples() << endl;
    cout << "Split : " << this->isIsSplit() << endl;
    cout << "Ratio : " << this->getRatio() << endl;
    cout << "threads : " << this->getThreads() << endl;
    cout << "workers : " << this->getWorkers() << endl;
    cout << "Normal Timing " << this->isIsNormalTime() << endl;
    cout << "Epoch Timing " << this->isIsEpochTime() << endl;
    cout << "Bulk : " << this->isBulk() << endl;
    cout << "BatchGapping : " << this->isBatch() << endl;
    cout << "Batch Gap : " << this->getBatch_per() <<endl;


}

OptArgs::OptArgs(const string &dataset, int features, int trainingSamples, int testingSamples, double alpha,
                 bool isSplit, double ratio, int threads, int workers, int iterations, bool isEpochTime,
                 bool isNormalTime) : dataset(dataset), features(features), trainingSamples(trainingSamples),
                                      testingSamples(testingSamples), alpha(alpha), isSplit(isSplit), ratio(ratio),
                                      threads(threads), workers(workers), iterations(iterations),
                                      isEpochTime(isEpochTime), isNormalTime(isNormalTime) {}

bool OptArgs::isIsEpochTime() const {
    return isEpochTime;
}

void OptArgs::setIsEpochTime(bool isEpochTime) {
    OptArgs::isEpochTime = isEpochTime;
}

bool OptArgs::isIsNormalTime() const {
    return isNormalTime;
}

void OptArgs::setIsNormalTime(bool isNormalTime) {
    OptArgs::isNormalTime = isNormalTime;
}

OptArgs::OptArgs(const string &dataset, int features, int trainingSamples, int testingSamples, double alpha,
                 bool isSplit, double ratio, int threads, int workers, int iterations, bool isEpochTime,
                 bool isNormalTime, bool bulk) : dataset(dataset), features(features), trainingSamples(trainingSamples),
                                                 testingSamples(testingSamples), alpha(alpha), isSplit(isSplit),
                                                 ratio(ratio), threads(threads), workers(workers),
                                                 iterations(iterations), isEpochTime(isEpochTime),
                                                 isNormalTime(isNormalTime), bulk(bulk) {}

bool OptArgs::isBulk() const {
    return bulk;
}

void OptArgs::setBulk(bool bulk) {
    OptArgs::bulk = bulk;
}

OptArgs::OptArgs(const string &dataset, int features, int trainingSamples, int testingSamples, double alpha,
                 bool isSplit, double ratio, int threads, int workers, int iterations, bool isEpochTime,
                 bool isNormalTime, bool bulk, bool batch, double batch_per) : dataset(dataset), features(features),
                                                                               trainingSamples(trainingSamples),
                                                                               testingSamples(testingSamples),
                                                                               alpha(alpha), isSplit(isSplit),
                                                                               ratio(ratio), threads(threads),
                                                                               workers(workers), iterations(iterations),
                                                                               isEpochTime(isEpochTime),
                                                                               isNormalTime(isNormalTime), bulk(bulk),
                                                                               batch(batch), batch_per(batch_per) {}

bool OptArgs::isBatch() const {
    return batch;
}

void OptArgs::setBatch(bool batch) {
    OptArgs::batch = batch;
}

double OptArgs::getBatch_per() const {
    return batch_per;
}

void OptArgs::setBatch_per(double batch_per) {
    OptArgs::batch_per = batch_per;
}
