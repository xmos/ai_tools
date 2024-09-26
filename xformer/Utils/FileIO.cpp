// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Utils/FileIO.h"
#include "Utils/TileRamSupport.h"

#include "mlir/Support/FileUtilities.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir::xcore::utils {

LogicalResult writeDataToFile(const std::string &filename, std::string data) {
  auto outputFile = openOutputFile(filename);
  if (!outputFile) {
    llvm::errs() << "Could not open output file: " << filename << "\n";
    return failure();
  }
  outputFile->os() << data;
  outputFile->keep();
  return success();
}

LogicalResult writeWeightsToFile(const std::string &filename,
                                 std::vector<std::vector<char>> tensorsVec,
                                 bool writeWeightsAsArray,
                                 bool placeInExternalMemory) {
  if (writeWeightsAsArray) {
    std::ostringstream cOut;
    cOut << R"(#include <stdint.h>)";

    if (placeInExternalMemory) {
      cOut << "\n\n"
           << R"(__attribute__ ((section(".ExtMem.data"))))"
           << "\n";
    } else {
      // Weights are to be placed in SRAM tile
      // Add tile ram server header
      auto tileHeader = utils::tileRamServerHeader();
      tensorsVec.insert(tensorsVec.begin(), tileHeader);
    }

    cOut << "const int8_t weights[] = {\n";
    int lineEnding = 0;
    int weightsSize = 0;
    for (auto const &tensor : tensorsVec) {
      for (auto const &i : tensor) {
        cOut << (int)i << ", ";
        lineEnding++;
        weightsSize++;
        if (lineEnding > 80) {
          cOut << "\n";
          lineEnding = 0;
        }
      }
    }

    cOut << R"(};
)";

    if (failed(utils::writeDataToFile(filename + ".c", cOut.str()))) {
      return failure();
    }

    std::ostringstream hOut;
    hOut << R"(#ifndef WEIGHTSGEN_H
#define WEIGHTSGEN_H

#define WEIGHTS_SIZE ()"
         << weightsSize << R"(U)

#endif // WEIGHTSGEN_H
)";

    return utils::writeDataToFile(filename + ".h", hOut.str());

  } else {
    // Write data for flash image
    // Combine data for the tensors
    std::string data;
    for (auto const &tensor : tensorsVec) {
      data += std::string(tensor.data(), tensor.size());
    }

    return utils::writeDataToFile(filename, data);
  }
}

LogicalResult getFlatBufferStringFromMLIR(
    mlir::ModuleOp module, std::map<std::string, std::string> metadata,
    const bool &dontMinify, std::string &flatBufferString) {
  std::unique_ptr<tensorflow::OpOrArgNameMapper> op_or_arg_name_mapper;
  if (dontMinify) {
    op_or_arg_name_mapper =
        std::make_unique<tensorflow::OpOrArgLocNameMapper>();
  } else {
    op_or_arg_name_mapper =
        std::make_unique<tensorflow::OpOrArgStripNameMapper>();
  }

  tflite::FlatbufferExportOptions options;
  bool emit_builtin_tflite_ops = true;
  bool emit_select_tf_ops = true;
  bool emit_custom_ops = true;

  options.toco_flags.set_force_select_tf_ops(!emit_builtin_tflite_ops);
  options.toco_flags.set_enable_select_tf_ops(emit_select_tf_ops);
  options.toco_flags.set_allow_custom_ops(emit_custom_ops);
  options.op_or_arg_name_mapper = op_or_arg_name_mapper.get();
  options.metadata = metadata;

  if (!tflite::MlirToFlatBufferTranslateFunction(module, options,
                                                 &flatBufferString)) {
    emitError(UnknownLoc::get(module.getContext()))
        << "Error converting MLIR to flatbuffer string!";
    return failure();
  }
  return success();
}

mlir::OwningOpRef<mlir::ModuleOp>
readFlatBufferFileToMLIR(const std::string &filename,
                         mlir::MLIRContext *context) {
  std::string errorMessage;
  auto inputFile = openInputFile(filename, &errorMessage);
  if (!inputFile) {
    emitError(UnknownLoc::get(context)) << errorMessage;
    return mlir::OwningOpRef<mlir::ModuleOp>(nullptr);
  }

  auto loc = mlir::FileLineColLoc::get(context, filename, 0, 0);
  mlir::OwningOpRef<mlir::ModuleOp> mod =
      tflite::FlatBufferToMlir(absl::string_view(inputFile->getBufferStart(),
                                                 inputFile->getBufferSize()),
                               context, loc);
  return mod;
}

} // namespace mlir::xcore::utils
