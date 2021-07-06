// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"
#include "Transforms/Passes.h"
#include "Utils/FileIO.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  static cl::opt<std::string> inputFilename(cl::Positional,
                                            cl::desc("<TFLite FlatBuffer>"));
  static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"));
  static cl::opt<bool> mlirIOEnabled("mlir-io",
                                     cl::desc("Enable MLIR input and output"));

  // Register any command line options.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  xcore::registerAllPasses();
  PassPipelineCLParser passPipeline("", "Compiler passes to run");
  cl::ParseCommandLineOptions(argc, argv);

  // Initialize dialects.
  MLIRContext context;
  context.loadDialect<StandardOpsDialect>();
  context.loadDialect<quant::QuantizationDialect>();
  context.loadDialect<TFL::TensorFlowLiteDialect>();
  context.loadDialect<xcore::XCoreDialect>();

  // Parse input.
  OwningModuleRef mod;
  if (mlirIOEnabled) {
    // Parse the MLIR input file.
    mod = parseSourceFile(inputFilename, &context);
    if (!mod) {
      llvm::errs() << "Unable to read MLIR file\n";
      return 1;
    }
  } else {
    // Read flatbuffer and convert to serialized MLIR string.
    mod = xcore::utils::readFlatBufferFileToMLIR(inputFilename, &context);
    if (!mod) {
      llvm::errs() << "Unable to read flatbuffer file\n";
      return 1;
    }
  }

  // Run transformations
  // Apply any pass manager command line options
  PassManager pm(&context, mlir::OpPassManager::Nesting::Implicit);
  applyPassManagerCLOptions(pm);

  auto errorHandler = [&](const Twine &msg) {
    return emitError(UnknownLoc::get(&context)) << msg;
  };

  if (passPipeline.hasAnyOccurrences()) {
    // Build the provided pipeline.
    if (failed(passPipeline.addToPipeline(pm, errorHandler)))
      return 1;

    // Run the pipeline.
    if (failed(pm.run(*mod)))
      return 1;

  } else {
    xcore::buildXCorePassPipeline(pm);
    if (failed(pm.run(*mod))) {
      llvm::errs() << "Running XCore pass pipeline failed\n";
      return 1;
    }
  }

  // Print output
  if (mlirIOEnabled) {
    // Print the MLIR output to stdout
    std::string errorMessage;
    auto output = openOutputFile("-", &errorMessage);
    if (!output) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }
    mod->print(output->os());
    output->os() << '\n';
  } else {
    // Write modified flatbuffer to output file
    if (!outputFilename.empty()) {
      std::string outfilename(outputFilename);
      if (failed(
              xcore::utils::writeMLIRToFlatBufferFile(outfilename, mod.get())))
        return 1;
    }
  }

  return 0;
}
