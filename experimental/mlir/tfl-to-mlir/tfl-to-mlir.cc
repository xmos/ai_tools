// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include <stddef.h>
#include <stdint.h>

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
// #include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
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

namespace mlir {

void registerTestPrintNestingPass() {
  PassRegistration<TestPrintNestingPass>("test-print-nesting",
                                         "Test various printing.");
}

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

  // OwningModuleRef attempt -- missing OpAsmPrinter
  // OpAsmPrinter p();
  // mod.get().print(p);

  // op builder attempt -- need to get model into builder
  // OpBuilder builder(&context);
  // for( Block & block : *builder.getBlock())
  //   printBlock(block);

  // Modify MLIR by building and running a PassManager
  PassManager pass_manager(&context, true);

  // pass_manager.addPass( std::make_unique<TestPrintNestingPass>() );
  pass_manager.addPass(std::make_unique<TestPrintNestingPass>());

  // TFL::QuantizationSpecs specs;
  // tensorflow::AddQuantizationPasses(specs, &pass_manager);

  pass_manager.run(mod.get());

  // Write modified flatbuffer
  std::string outfilename(filename + ".out");
  write_mlir_to_flatbuffer(outfilename, mod.get());

  return 0;
}
}  // end namespace mlir