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
    cout << "Drop : " << this->isIsDrop() << endl;
    cout << "Drop Percentage : " << this->getDrop_out_per() << endl;
    cout << "Sequential : " << this->isSequential() << endl;
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

OptArgs::OptArgs(const string &dataset, int features, int trainingSamples, int testingSamples, double alpha,
                 bool isSplit, double ratio, int threads, int workers, int iterations, bool isEpochTime,
                 bool isNormalTime, bool bulk, bool batch, double batch_per, double drop_out_per, bool isDrop)
        : dataset(dataset), features(features), trainingSamples(trainingSamples), testingSamples(testingSamples),
          alpha(alpha), isSplit(isSplit), ratio(ratio), threads(threads), workers(workers), iterations(iterations),
          isEpochTime(isEpochTime), isNormalTime(isNormalTime), bulk(bulk), batch(batch), batch_per(batch_per),
          drop_out_per(drop_out_per), isDrop(isDrop) {}

double OptArgs::getDrop_out_per() const {
    return drop_out_per;
}

void OptArgs::setDrop_out_per(double drop_out_per) {
    OptArgs::drop_out_per = drop_out_per;
}

bool OptArgs::isIsDrop() const {
    return isDrop;
}

void OptArgs::setIsDrop(bool isDrop) {
    OptArgs::isDrop = isDrop;
}

OptArgs::OptArgs(const string &dataset, int features, int trainingSamples, int testingSamples, double alpha,
                 bool isSplit, double ratio, int threads, int workers, int iterations, bool isEpochTime,
                 bool isNormalTime, bool bulk, bool batch, double batch_per, double drop_out_per, bool isDrop,
                 bool sequential) : dataset(dataset), features(features), trainingSamples(trainingSamples),
                                    testingSamples(testingSamples), alpha(alpha), isSplit(isSplit), ratio(ratio),
                                    threads(threads), workers(workers), iterations(iterations),
                                    isEpochTime(isEpochTime), isNormalTime(isNormalTime), bulk(bulk), batch(batch),
                                    batch_per(batch_per), drop_out_per(drop_out_per), isDrop(isDrop),
                                    sequential(sequential) {}

bool OptArgs::isSequential() const {
    return sequential;
}

void OptArgs::setSequential(bool sequential) {
    OptArgs::sequential = sequential;
}

OptArgs::OptArgs(const string &dataset, int features, int trainingSamples, int testingSamples, double alpha,
                 bool isSplit, double ratio, int threads, int workers, int iterations, bool isEpochTime,
                 bool isNormalTime, bool bulk, bool batch, double batch_per, double drop_out_per, bool isDrop,
                 bool sequential, bool ring, bool randomringv1, bool randomringv2) : dataset(dataset),
                                                                                     features(features),
                                                                                     trainingSamples(trainingSamples),
                                                                                     testingSamples(testingSamples),
                                                                                     alpha(alpha), isSplit(isSplit),
                                                                                     ratio(ratio), threads(threads),
                                                                                     workers(workers),
                                                                                     iterations(iterations),
                                                                                     isEpochTime(isEpochTime),
                                                                                     isNormalTime(isNormalTime),
                                                                                     bulk(bulk), batch(batch),
                                                                                     batch_per(batch_per),
                                                                                     drop_out_per(drop_out_per),
                                                                                     isDrop(isDrop),
                                                                                     sequential(sequential), ring(ring),
                                                                                     randomringv1(randomringv1),
                                                                                     randomringv2(randomringv2) {}

bool OptArgs::isRing() const {
    return ring;
}

void OptArgs::setRing(bool ring) {
    OptArgs::ring = ring;
}

bool OptArgs::isRandomringv1() const {
    return randomringv1;
}

void OptArgs::setRandomringv1(bool randomringv1) {
    OptArgs::randomringv1 = randomringv1;
}

bool OptArgs::isRandomringv2() const {
    return randomringv2;
}

void OptArgs::setRandomringv2(bool randomringv2) {
    OptArgs::randomringv2 = randomringv2;
}

OptArgs::OptArgs(const string &dataset, int features, int trainingSamples, int testingSamples, double alpha,
                 bool isSplit, double ratio, int threads, int workers, int iterations, bool isEpochTime,
                 bool isNormalTime, bool bulk, bool batch, double batch_per, double drop_out_per, bool isDrop,
                 bool sequential, bool ring, bool randomringv1, bool randomringv2, bool fullbatchv1) : dataset(dataset),
                                                                                                       features(
                                                                                                               features),
                                                                                                       trainingSamples(
                                                                                                               trainingSamples),
                                                                                                       testingSamples(
                                                                                                               testingSamples),
                                                                                                       alpha(alpha),
                                                                                                       isSplit(isSplit),
                                                                                                       ratio(ratio),
                                                                                                       threads(threads),
                                                                                                       workers(workers),
                                                                                                       iterations(
                                                                                                               iterations),
                                                                                                       isEpochTime(
                                                                                                               isEpochTime),
                                                                                                       isNormalTime(
                                                                                                               isNormalTime),
                                                                                                       bulk(bulk),
                                                                                                       batch(batch),
                                                                                                       batch_per(
                                                                                                               batch_per),
                                                                                                       drop_out_per(
                                                                                                               drop_out_per),
                                                                                                       isDrop(isDrop),
                                                                                                       sequential(
                                                                                                               sequential),
                                                                                                       ring(ring),
                                                                                                       randomringv1(
                                                                                                               randomringv1),
                                                                                                       randomringv2(
                                                                                                               randomringv2),
                                                                                                       fullbatchv1(
                                                                                                               fullbatchv1) {}


bool OptArgs::isFullbatchv1() const {
    return fullbatchv1;
}

void OptArgs::setFullbatchv1(bool fullbatchv1) {
    OptArgs::fullbatchv1 = fullbatchv1;
}
