// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Utils/FileIO.h"

#include "flatbuffers/flexbuffers.h"
#include "mlir/Support/FileUtilities.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir {
namespace xcore {
namespace utils {

LogicalResult writeDataToFile(std::string &filename, std::string &data) {
  auto outputFile = openOutputFile(filename);
  if (!outputFile) {
    llvm::errs() << "Could not open output file: " << filename << "\n";
    return failure();
  }
  outputFile->os() << data;
  outputFile->keep();
  return success();
}

LogicalResult writeFlashImageToFile(std::string &filename,
                                    std::vector<std::vector<char>> tensorsVec) {
  // Create flexbuffer of params
  flexbuffers::Builder fbb;
  auto rootMap = fbb.StartMap();
  auto paramsVec = fbb.StartVector("params");
  // For each tensor, we add a new flexbuffer blob
  for (auto const &tensor : tensorsVec) {
    fbb.Blob(tensor.data(), tensor.size());
  }
  fbb.EndVector(paramsVec, false, false);
  fbb.EndMap(rootMap);
  fbb.Finish();

  // Write flexbuffer data to file
  std::string fbbData(fbb.GetBuffer().begin(), fbb.GetBuffer().end());
  return utils::writeDataToFile(filename, fbbData);
}

LogicalResult writeMLIRToFlatBufferFile(std::string &filename,
                                        mlir::ModuleOp module) {
  std::string serialized_flatbuffer;
  tflite::FlatbufferExportOptions options;
  options.emit_builtin_tflite_ops = true;
  options.emit_select_tf_ops = true;
  options.emit_custom_ops = true;

  if (tflite::MlirToFlatBufferTranslateFunction(module, options,
                                                &serialized_flatbuffer)) {
    return writeDataToFile(filename, serialized_flatbuffer);
  } else {
    llvm::errs() << "Error converting MLIR to flatbuffer, no file written"
                 << "\n";
    return failure();
  }
}

mlir::OwningModuleRef readFlatBufferFileToMLIR(std::string &filename,
                                               mlir::MLIRContext *context) {
  std::string errorMessage;
  auto inputFile = openInputFile(filename, &errorMessage);
  if (!inputFile) {
    llvm::errs() << errorMessage << "\n";
    return mlir::OwningModuleRef(nullptr);
  }

  auto loc = mlir::FileLineColLoc::get(context, filename, 0, 0);
  OwningModuleRef mod =
      tflite::FlatBufferToMlir(absl::string_view(inputFile->getBufferStart(),
                                                 inputFile->getBufferSize()),
                               context, loc);
  return mod;
}

} // namespace utils
} // namespace xcore
} // namespace mlir
