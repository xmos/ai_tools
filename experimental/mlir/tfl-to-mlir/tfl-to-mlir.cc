// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <fstream>
#include <iostream>
#include <string>

#include "file_io.hpp"
#include "flatbuffer_to_string.hpp"
#include "TestPrintNestingPass.hpp"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"

using namespace mlir;

int main(int argc, char *argv[]) {
  std::cout << argc << "\t" << argv[1] << std::endl;
  std::string filename;
  if (argc > 1) {
    filename = argv[1];
  } else {
    std::cout << "Expected filename argument" << std::endl;
    return 1;
  }

  MLIRContext context;

  // Read flatbuffer and convert to serialized MLIR string
  std::string serialized_model;

  OwningModuleRef mod(
      read_flatbuffer_to_mlir(filename, serialized_model, &context));
  if (!mod) {
    std::cout << "Unable to read flatbuffer" << std::endl;
    return 1;
  }

  // Modify MLIR by building and running a PassManager
  PassManager pass_manager(&context, true);

  // For more info on pass management:
  // https://github.com/llvm/llvm-project/blob/e79cd47e1620045562960ddfe17ab0c4f6e6628f/mlir/docs/PassManagement.md
  pass_manager.addPass(std::make_unique<local::TestPrintNestingPass>());

  // Add additional passes:
  // TFL::QuantizationSpecs specs;
  // tensorflow::AddQuantizationPasses(specs, &pass_manager);

  pass_manager.run(mod.get());

  // Write modified flatbuffer
  std::string outfilename(filename + ".out");
  write_mlir_to_flatbuffer(outfilename, mod.get());

  return 0;
}
