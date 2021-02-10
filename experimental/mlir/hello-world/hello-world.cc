// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1
#include "lib-test.h"
#include "mlir/IR/Dialect.h"
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

    mlir::MLIRContext context;

    // read
    mlir::OwningModuleRef module(mlir_read_flatbuffer( &context, filename ));
    if (!module) return 1;

    // modify


    //write
    std::string outfilename("test.tflite");
    mlir_write_flatbuffer( outfilename, module.get());

    std::cout << "Some stuff " << M_PI << std::endl;
    return 0;
}
