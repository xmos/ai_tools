// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#ifndef XFORMER_UTILS_FILEIO_H
#define XFORMER_UTILS_FILEIO_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/FileUtilities.h"

namespace mlir {
namespace xcore {
namespace utils {

LogicalResult writeMLIRToFlatBufferFile(std::string &filename,
                                        mlir::ModuleOp module);

mlir::OwningModuleRef readFlatBufferFileToMLIR(std::string &filename,
                                               mlir::MLIRContext *context);

} // namespace utils
} // namespace xcore
} // namespace mlir

#endif // XFORMER_UTILS_FILEIO_H
