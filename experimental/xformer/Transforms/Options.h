// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#ifndef XFORMER_TRANSFORMS_OPTIONS_H
#define XFORMER_TRANSFORMS_OPTIONS_H

#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace xcore {

extern llvm::cl::opt<std::string> flashImageFilenameOption;

} // namespace xcore
} // namespace mlir

#endif // XFORMER_TRANSFORMS_OPTIONS_H
