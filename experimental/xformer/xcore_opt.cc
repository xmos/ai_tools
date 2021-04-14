// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "transforms/passes.h"
#include "utils/file_io.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  static cl::opt<std::string> inputPath(
      cl::Positional, cl::desc("<TFLite FlatBuffer>"), cl::Required);
  static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"));

  // Register any command line options.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  xcore::registerAllPasses();
  PassPipelineCLParser passPipeline("", "Compiler passes to run");
  cl::ParseCommandLineOptions(argc, argv);

  // Initialize dialects.
  DialectRegistry registry;
  // registry.insert<TFL::TensorFlowLiteDialect>();
  // registry.insert<TF::TensorFlowDialect>();
  // registry.insert<quant::QuantizationDialect>();
  // registry.insert<StandardOpsDialect>();

  // Convert the Module proto into MLIR.
  MLIRContext context; //(registry);
  // context.loadAllAvailableDialects();

  // Read flatbuffer and convert to serialized MLIR string
  OwningModuleRef mod(
      xcore::utils::readFlatBufferFileToMLIR(inputPath, &context));
  if (!mod) {
    llvm::errs() << "Unable to read flatbuffer\n";
    return 1;
  }

  // Run transformations.
  // Apply any pass manager command line options.
  PassManager pm(&context);
  applyPassManagerCLOptions(pm);

  if (passPipeline.hasAnyOccurrences()) {
    // Build the provided pipeline.
    if (failed(passPipeline.addToPipeline(pm)))
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

  // Write modified flatbuffer
  if (!outputFilename.empty()) {
    std::string outfilename(outputFilename);
    if (failed(xcore::utils::writeMLIRToFlatBufferFile(outfilename, mod.get())))
      return 1;
  }

  return 0;
}
