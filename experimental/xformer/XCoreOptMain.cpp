// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"
#include "Transforms/Options.h"
#include "Transforms/Passes.h"
#include "Utils/FileIO.h"

#include "mlir/IR/AsmState.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;

// TODO:
namespace mlir {
namespace xcore {

cl::opt<unsigned> threadCountOption("xcore-thread-count",
                                    cl::desc("Thread count"), cl::init(1));

cl::opt<std::string> flashImageFilenameOption(
    "xcore-flash-image-file",
    cl::desc("The file to write the xcore flash image."),
    cl::value_desc("filename"), cl::init(""));

cl::opt<unsigned> loadExternallyIfLargerOption(
    "xcore-load-externally-if-larger",
    cl::desc("Load constants externally if larger than given limit in bytes "
             "(default = 96 bytes). Cannot be specified when "
             "xcore-flash-image-file is not provided."),
    cl::init(96));

cl::opt<bool> reduceMemoryOption(
    "xcore-reduce-memory",
    cl::desc(
        "Try to reduce memory usage by possibly increasing execution time."),
    cl::init(true));

} // namespace xcore
} // namespace mlir

LogicalResult runPassPipeline(const PassPipelineCLParser &passPipeline,
                              const OwningModuleRef &mod,
                              MLIRContext *context) {
  PassManager pm(context, mlir::OpPassManager::Nesting::Implicit);
  applyPassManagerCLOptions(pm);

  auto errorHandler = [&](const Twine &msg) {
    return emitError(UnknownLoc::get(context)) << msg;
  };

  if (passPipeline.hasAnyOccurrences()) {
    // Build the provided pipeline.
    if (failed(passPipeline.addToPipeline(pm, errorHandler)))
      return failure();

    // Run the pipeline.
    if (failed(pm.run(*mod)))
      return failure();

  } else {
    xcore::buildXCorePassPipeline(pm);
    if (failed(pm.run(*mod))) {
      return failure();
    }
  }
  return success();
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  static cl::opt<std::string> inputFilename(cl::Positional,
                                            cl::desc("<TFLite FlatBuffer>"));
  static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"));
  static cl::opt<bool> mlirIOEnabled(
      "mlir-io", cl::desc("Enable MLIR input and output"), cl::init(false));
  static cl::opt<bool> verifyDiagnosticsEnabled(
      "verify-diagnostics",
      cl::desc("Check that emitted diagnostics match "
               "expected-* lines on the corresponding line"),
      cl::init(false));
  static cl::opt<bool> dontMinifyEnabled(
      "xcore-dont-minify",
      cl::desc("Do not strip debug info and minify the model"),
      cl::init(false));

  // Register any command line options.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  xcore::registerXCorePassPipeline();
  PassPipelineCLParser passPipeline("", "Compiler passes to run");
  cl::ParseCommandLineOptions(argc, argv);

  // Initialize dialects.
  MLIRContext context;
  context.loadDialect<StandardOpsDialect>();
  context.loadDialect<quant::QuantizationDialect>();
  context.loadDialect<TFL::TensorFlowLiteDialect>();
  context.loadDialect<xcore::XCoreDialect>();
  context.printOpOnDiagnostic(!verifyDiagnosticsEnabled);

  auto failedMessage = [&](const Twine &msg) {
    emitError(UnknownLoc::get(&context)) << msg;
    return 1;
  };

  // Validate options
  if (mlir::xcore::loadExternallyIfLargerOption.getNumOccurrences() > 0 &&
      mlir::xcore::flashImageFilenameOption.empty()) {
    return failedMessage(
        "Please specify the xcore-flash-image-file option when specifying the "
        "xcore-load-externally-if-larger option!");
  }

  if (mlir::xcore::threadCountOption > 5) {
    return failedMessage(
        "Please specify a thread count between one and five!");
  }

  // Parse input.
  OwningModuleRef mod;
  SourceMgr sourceMgr;
  if (mlirIOEnabled) {
    // Parse the MLIR input file.
    std::string errorMessage;
    auto file = openInputFile(inputFilename, &errorMessage);
    if (!file) {
      return failedMessage(errorMessage);
    }
    sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
    mod = parseSourceFile(sourceMgr, &context);
  } else {
    // Read flatbuffer and convert to serialized MLIR string.
    mod = xcore::utils::readFlatBufferFileToMLIR(inputFilename, &context);
    if (!mod) {
      return failedMessage("Unable to read flatbuffer file!");
    }
  }

  // Run transformations
  if (verifyDiagnosticsEnabled) {
    SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
    (void)runPassPipeline(passPipeline, mod, &context);
    if (failed(sourceMgrHandler.verify())) {
      return 1;
    }
  } else {
    if (failed(runPassPipeline(passPipeline, mod, &context))) {
      return 1;
    }
  }

  // Print output
  if (mlirIOEnabled) {
    // Print the MLIR output to stdout
    std::string errorMessage;
    auto output = openOutputFile("-", &errorMessage);
    if (!output) {
      return failedMessage(errorMessage);
    }
    mod->print(output->os());
    output->os() << '\n';
  }
  // Write modified flatbuffer to output file
  if (!outputFilename.empty()) {
    std::string outfilename(outputFilename);
    if (failed(xcore::utils::writeMLIRToFlatBufferFile(outfilename, mod.get(),
                                                       dontMinifyEnabled)))
      return 1;
  }

  return 0;
}
