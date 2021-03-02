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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace { 
/// @note: this scope was lifted from: 
/// https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/
/// and source files in llvm-project repo

// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// This pass illustrates the IR nesting through printing.
struct TestPrintNestingPass
    : public PassWrapper<TestPrintNestingPass, OperationPass<>> {
  // Entry point for the pass.
  void runOnOperation() override {
    Operation *op = getOperation();
    resetIndent();
    printOperation(op);
  }

  /// The three methods below are mutually recursive and follow the nesting of
  /// the IR: operation->region->block->operation->...

  void printOperation(Operation *op) {
    // Print the operation itself and some of its properties
    printIndent() << "visiting op: '" << op->getName() << "' with "
                  << op->getNumOperands() << " operands and "
                  << op->getNumResults() << " results\n";
    // Print the operation attributes
    if (!op->getAttrs().empty()) {
      printIndent() << op->getAttrs().size() << " attributes:\n";
      for (NamedAttribute attr : op->getAttrs())
        printIndent() << " - '" << attr.first << "' : '" << attr.second
                      << "'\n";
    }

    // Recurse into each of the regions attached to the operation.
    printIndent() << " " << op->getNumRegions() << " nested regions:\n";
    auto indent = pushIndent();
    for (Region &region : op->getRegions()) printRegion(region);
  }

  void printRegion(Region &region) {
    // A region does not hold anything by itself other than a list of blocks.
    printIndent() << "Region with " << region.getBlocks().size()
                  << " blocks:\n";
    auto indent = pushIndent();
    for (Block &block : region.getBlocks()) printBlock(block);
  }

  void printBlock(Block &block) {
    // Print the block intrinsics properties (basically: argument list)
    printIndent()
        << "Block with " << block.getNumArguments() << " arguments, "
        << block.getNumSuccessors()
        << " successors, and "
        // Note, this `.size()` is traversing a linked-list and is O(n).
        << block.getOperations().size() << " operations\n";

    // Block main role is to hold a list of Operations: let's recurse.
    auto indent = pushIndent();
    for (Operation &op : block.getOperations()) printOperation(&op);
  }

  /// Manages the indentation as we traverse the IR nesting.
  int indent;
  struct IdentRAII {
    int &indent;
    IdentRAII(int &indent) : indent(indent) {}
    ~IdentRAII() { --indent; }
  };
  void resetIndent() { indent = 0; }
  IdentRAII pushIndent() { return IdentRAII(++indent); }

  llvm::raw_ostream &printIndent() {
    for (int i = 0; i < indent; ++i) llvm::outs() << "  ";
    return llvm::outs();
  }
};
}  // end anonymous namespace

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
  pass_manager.addPass(std::make_unique<TestPrintNestingPass>());

  // Add additional passes:
  // TFL::QuantizationSpecs specs;
  // tensorflow::AddQuantizationPasses(specs, &pass_manager);

  pass_manager.run(mod.get());

  // Write modified flatbuffer
  std::string outfilename(filename + ".out");
  write_mlir_to_flatbuffer(outfilename, mod.get());

  return 0;
}
