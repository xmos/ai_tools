
#ifndef LIB_TEST_H
#define LIB_TEST_H

// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "lib-test.h"
#include <math.h>

#include <iostream>
#include <fstream> 

#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
// #include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"


mlir::OwningModuleRef mlir_read_flatbuffer(mlir::MLIRContext* context, std::string & filename ) // TODO remove static once this is working
{

    std::ifstream infile;
    infile.open(filename, std::ios::binary | std::ios::in);
    infile.seekg(0,std::ios::end);
    int length = infile.tellg();
    infile.seekg(0,std::ios::beg);
    char *data = new char[length];
    infile.read(data, length);
    infile.close();


    return tflite::FlatBufferToMlir(data, context, mlir::UnknownLoc::get(context));

}

static void mlir_write_flatbuffer() // TODO remove static once this is working
{
    
}

void print_some_stuff()
{
    mlir::MLIRContext * context = new mlir::MLIRContext();
    std::string filename = "../tflite2xcore/tflite2xcore/tests/test_ir/builtin_operators.tflite";
    mlir_read_flatbuffer( context, filename );

    std::cout << "Some stuff " << M_PI << std::endl;
}

#endif /* LIB_TEST_H */ 
