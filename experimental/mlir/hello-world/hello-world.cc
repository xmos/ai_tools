// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1
#include "lib-test.h"
#include <iostream>
int main(int argc, char *argv[])
{
    std::cout << argc << "\t" << argv[1] << std::endl;
    std::string filename;
    if(argc>1){
        filename = argv[1];
    }else{
        filename = "../tflite2xcore/tflite2xcore/tests/test_ir/builtin_operators.tflite";
    }

    print_some_stuff(filename);
    return 0;
}
