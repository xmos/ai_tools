// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#ifndef XFORMER_TRANSFORMS_OPTIONS_H
#define XFORMER_TRANSFORMS_OPTIONS_H

#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace xcore {

extern llvm::cl::opt<unsigned> threadCountOption;
extern llvm::cl::opt<std::string> flashImageFilenameOption;
extern llvm::cl::opt<unsigned> loadExternallyIfLargerOption;
extern llvm::cl::opt<double> convQuantErrorThresholdOption;
extern llvm::cl::opt<bool> convForceErrorCheckOption;
extern llvm::cl::opt<unsigned> convMultiplierFactorOption;
extern llvm::cl::opt<bool> opSplitTensorArenaOption;
extern llvm::cl::list<int32_t> opSplitStartOpOption;
extern llvm::cl::list<int32_t> opSplitEndOpOption;
extern llvm::cl::list<int32_t> opSplitNumSplitsOption;
extern llvm::cl::opt<bool> allowInputModificationOption;
extern llvm::cl::opt<bool> convDebugOption;

} // namespace xcore
} // namespace mlir

#endif // XFORMER_TRANSFORMS_OPTIONS_H
