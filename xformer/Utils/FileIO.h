// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#ifndef XFORMER_UTILS_FILEIO_H
#define XFORMER_UTILS_FILEIO_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir::xcore::utils {

LogicalResult writeDataToFile(const std::string &filename, std::string data);

LogicalResult writeWeightsToFile(const std::string &filename,
                                 std::vector<std::vector<char>> tensorsVec);

LogicalResult
writeTileServerDataToFile(const std::string &filename,
                          std::vector<std::vector<char>> tensorsVec,
                          bool placeInExternalMemory);

LogicalResult getFlatBufferStringFromMLIR(
    mlir::ModuleOp module, std::map<std::string, std::string> metadata,
    const bool &dontMinify, std::string &flatBufferString);

mlir::OwningOpRef<mlir::ModuleOp>
readFlatBufferFileToMLIR(const std::string &filename,
                         mlir::MLIRContext *context);

} // namespace mlir::xcore::utils

#endif // XFORMER_UTILS_FILEIO_H
