// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#ifndef XFORMER_UTILS_DIAGNOSTICS_H
#define XFORMER_UTILS_DIAGNOSTICS_H

#include <string>

namespace mlir::xcore::utils {

template <typename T> std::string getMsgWithLocPrefix(T &op, std::string msg) {
  std::string message;
  llvm::raw_string_ostream os(message);
  if (!op.getLoc().template isa<UnknownLoc>()) {
    os << op.getLoc() << " : ";
  }
  os << msg;
  return message;
}

} // namespace mlir::xcore::utils

#endif // XFORMER_UTILS_DIAGNOSTICS_H
