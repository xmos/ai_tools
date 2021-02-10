// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1
#ifndef LIB_TEST_H
#define LIB_TEST_H

#include <string>

#include "absl/strings/string_view.h"
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project

mlir::OwningModuleRef mlir_read_flatbuffer(mlir::MLIRContext* context, std::string & filename );

void mlir_write_flatbuffer( std::string & filename, mlir::ModuleOp module);


#endif /* LIB_TEST_H */ 

