// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Utils/Diagnostics.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Location.h"

namespace mlir {
namespace xcore {
namespace utils {

template<typename T>
std::string getMsgWithLocPrefix(T &op, std::string msg) {
    std::string message;
    llvm::raw_string_ostream os(message);
    if (!op.getLoc().template isa<UnknownLoc>()) {
      os << op.getLoc() << " : ";
    }
    os << msg;
    return message;
  };

} // namespace utils
} // namespace xcore
} // namespace mlir
