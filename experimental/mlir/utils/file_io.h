#ifndef XCORE_UTILS_FILE_IO_H
#define XCORE_UTILS_FILE_IO_H

#include "mlir/Support/FileUtilities.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir {
namespace xcore {
namespace utils {

LogicalResult writeMLIRToFlatBufferFile(std::string &filename,
                                        mlir::ModuleOp module) {
  std::string serialized_flatbuffer;

  if (!tflite::MlirToFlatBufferTranslateFunction(module, &serialized_flatbuffer,
                                                 true, true, true)) {
    auto outputFile = openOutputFile(filename);
    if (!outputFile) {
      llvm::errs() << "Could not open output file: " << filename << "\n";
      return failure();
    }

    outputFile->os() << serialized_flatbuffer;
    outputFile->keep();
    return success();
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

  auto loc = mlir::FileLineColLoc::get(inputFile->getBufferIdentifier(), 0, 0,
                                       context);
  OwningModuleRef mod =
      tflite::FlatBufferToMlir(absl::string_view(inputFile->getBufferStart(),
                                                 inputFile->getBufferSize()),
                               context, loc);
  return mod;
}

} // namespace utils
} // namespace xcore
} // namespace mlir

#endif // XCORE_UTILS_FILE_IO_H
