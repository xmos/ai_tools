// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "lib-test.h"
#include <cstdio>
#include <math.h>

#include <iostream>
#include <fstream> 

#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
// #include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"


mlir::OwningModuleRef mlir_read_flatbuffer(mlir::MLIRContext* context, std::string & filename )
{

    std::ifstream infile;
    infile.open(filename, std::ios::binary | std::ios::in);
    infile.seekg(0,std::ios::end);
    int length = infile.tellg();
    infile.seekg(0,std::ios::beg);
    char *buffer = new char[length];
    infile.read(buffer, length);

    if (infile)
      std::cout << "all " << length << " characters read successfully." << std::endl;
    else
      std::cout << "error: only " << infile.gcount() << " could be read" << std::endl;
    infile.close();


    return tflite::FlatBufferToMlir(buffer, context, mlir::UnknownLoc::get(context));

}

void mlir_write_flatbuffer( std::string & filename, mlir::ModuleOp module)
{

    std::string * serialized_flatbuffer = new std::string();
    serialized_flatbuffer->resize(1000000); // TODO figure out what this size should be, or at least a tigher bound
    std::cout << *serialized_flatbuffer << std::endl;

    if(!tflite::MlirToFlatBufferTranslateFunction(  module, serialized_flatbuffer,
                                            true, true, true)){

        std::ofstream outfile (filename,std::ofstream::binary);
        outfile.write (serialized_flatbuffer->data(),serialized_flatbuffer->size());
        outfile.close();

    } else {
        std::cout << "Error converting MLIR to flatbuffer, no file written" << std::endl;
    }
    delete serialized_flatbuffer;
}
