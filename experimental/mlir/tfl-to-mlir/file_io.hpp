#ifndef FILE_IO_HPP
#define FILE_IO_HPP

#include "flatbuffer_to_string.hpp"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/tf_tfl_passes.h"

void write_mlir_to_flatbuffer(std::string& filename, mlir::ModuleOp module) {
  std::string serialized_flatbuffer;

  if (!tflite::MlirToFlatBufferTranslateFunction(module, &serialized_flatbuffer,
                                                 true, true, true)) {
    std::ofstream outfile(filename, std::ofstream::binary);
    outfile.write(serialized_flatbuffer.data(), serialized_flatbuffer.size());
    outfile.close();

  } else {
    std::cout << "Error converting MLIR to flatbuffer, no file written"
              << std::endl;
  }
}

mlir::OwningModuleRef read_flatbuffer_to_mlir(std::string& filename,
                                              std::string& serialized_model,
                                              mlir::MLIRContext* context) {
  if (tflite::ReadAndVerify(filename, &serialized_model)) {
    return mlir::OwningModuleRef(nullptr);
  }

  tflite::ToString(serialized_model);
  mlir::OwningModuleRef mod(tflite::FlatBufferToMlir(
      serialized_model, context, mlir::UnknownLoc::get(context)));
  return mod;
}

#endif  // FILE_IO_HPP