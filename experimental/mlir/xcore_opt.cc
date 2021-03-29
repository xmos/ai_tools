#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
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
                                             cl::value_desc("filename"),
                                             cl::init("-"));

  // Register any command line options.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  mlir::iree_integrations::TFL::registerAllPasses();
  mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run");
  cl::ParseCommandLineOptions(argc, argv);

  // Initialize dialects.
  DialectRegistry registry;
  // registry.insert<TFL::TensorFlowLiteDialect>();
  // registry.insert<TF::TensorFlowDialect>();
  // registry.insert<quant::QuantizationDialect>();
  // registry.insert<StandardOpsDialect>();

  // Convert the Module proto into MLIR.
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  /*
  // Load input buffer.
  std::string errorMessage;
  auto inputFile = openInputFile(inputPath, &errorMessage);
  if (!inputFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }
  */

  // Read flatbuffer and convert to serialized MLIR string
  std::string serialized_model;
  OwningModuleRef mod(
      read_flatbuffer_to_mlir(inputPath, serialized_model, &context));
  if (!mod) {
    std::cout << "Unable to read flatbuffer" << std::endl;
    return 1;
  }

  /*
  // Save.
  auto saveToFile = [&](llvm::StringRef savePath) -> LogicalResult {
    auto outputFile = openOutputFile(savePath);
    if (!outputFile) {
      llvm::errs() << "Could not open output file: " << savePath << "\n";
      return failure();
    }
    OpPrintingFlags printFlags;
    module->print(outputFile->os(), printFlags);
    outputFile->os() << "\n";
    outputFile->keep();
    return success();
  };

  // Save temp input.
  if (!saveTempTflInput.empty()) {
    if (failed(saveToFile(saveTempTflInput))) return 10;
  }*/

  // Run transformations.
  // Apply any pass manager command line options.
  PassManager pm(context, OpPassManager::Nesting::Implicit);
  applyPassManagerCLOptions(pm);

  auto errorHandler = [&](const Twine &msg) {
    emitError(UnknownLoc::get(context)) << msg;
    return failure();
  };

  if (passPipeline.hasAnyOccurrences()) {
    // Build the provided pipeline.
    if (failed(passPipeline.addToPipeline(pm, errorHandler)))
      return failure();

    // Run the pipeline.
    if (failed(pm.run(*mod)))
      return failure();

  } else {
    mlir::iree_integrations::TFL::buildTFLImportPassPipeline(pm);
    if (failed(pm.run(*mod))) {
      llvm::errs() << "Running iree-import-tflite pass pipeline failed (see "
                      "diagnostics)\n";
      return 3;
    }
  }

  /*
  // Save temp output.
  if (!saveTempIreeImport.empty()) {
    if (failed(saveToFile(saveTempIreeImport))) return 10;
  }

  // Save output.
  if (failed(saveToFile(outputFilename))) return 3;
  */

  // Write modified flatbuffer
  std::string outfilename(filename + ".out");
  write_mlir_to_flatbuffer(outfilename, mod.get());

  return 0;
}
