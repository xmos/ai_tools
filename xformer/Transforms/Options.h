// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#ifndef XFORMER_TRANSFORMS_OPTIONS_H
#define XFORMER_TRANSFORMS_OPTIONS_H

#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace xcore {

extern llvm::cl::opt<unsigned> quadraticLookupErrorOption;
extern llvm::cl::opt<bool> enableBetaFloatOption;
extern llvm::cl::opt<unsigned> threadCountOption;
extern llvm::cl::opt<std::string> weightsFilenameOption;
extern llvm::cl::opt<unsigned> loadExternallyIfLargerOption;
extern llvm::cl::opt<bool> weightsAsArrayOption;
extern llvm::cl::opt<bool> weightsInExternalMemory;
extern llvm::cl::opt<unsigned> maxLoadExternalSizeOption;
extern llvm::cl::opt<double> convQuantErrorThresholdOption;
extern llvm::cl::opt<bool> convForceErrorCheckOption;
extern llvm::cl::opt<unsigned> convMultiplierFactorOption;
extern llvm::cl::opt<bool> opSplitTensorArenaOption;
extern llvm::cl::opt<unsigned> opSplitTargetSizeOption;
extern llvm::cl::list<unsigned> opSplitBottomOpsOption;
extern llvm::cl::list<unsigned> opSplitTopOpsOption;
extern llvm::cl::list<unsigned> opSplitNumSplitsOption;
extern llvm::cl::opt<bool> allowInputModificationOption;
extern llvm::cl::opt<bool> mergeTransposeOption;
extern llvm::cl::opt<bool> convDebugOption;
extern llvm::cl::opt<bool> overlapConvOption;
extern llvm::cl::opt<bool> offlineOffsetsOption;
extern llvm::cl::opt<unsigned> convChannelwiseSplitSizeOption;
} // namespace xcore
} // namespace mlir

#endif // XFORMER_TRANSFORMS_OPTIONS_H
