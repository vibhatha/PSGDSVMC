#include <iostream>
#include "Test.h"
#include "DataSet.h"

using namespace std;

void test1();
void test2();

int main() {
    std::cout << "Hello, World!" << std::endl;

    test2();

    return 0;
}

void test1(){
    Test test;

    test.test1();
}

void test2() {
    Test test;
    test.test2();
}