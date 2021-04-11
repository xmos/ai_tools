#ifndef XCORE_UTILS_FILE_IO_H
#define XCORE_UTILS_FILE_IO_H

#include "mlir/IR/Module.h"
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

#endif // XCORE_UTILS_FILE_IO_H
