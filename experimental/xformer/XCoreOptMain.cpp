// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"
#include "Transforms/Options.h"
#include "Transforms/Passes.h"
#include "Utils/FileIO.h"
#include "Version.h"

#include "lib_nn/api/version.h"
#include "lib_tflite_micro/api/version.h"
#include "lib_tflite_micro/api/xcore_shared_config.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
// TODO: dpk
// refactor tflmc to have include folder
#include "src/Compiler.h"

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

// This option is to provide an error threshold.
// The maximum average error between the reference and quantised
// implementations of the output transform over each channel is used to decide
// if to replace TFL Conv ops with XC Conv ops.
// The average is defined over the range of non-saturating
// accumulators, i.e. accumulators that do not reach a saturating output in the
// int8 space. The error calcualated is the maximum average for all of the
// channels.
cl::opt<double> convQuantErrorThresholdOption(
    "xcore-conv-err-threshold",
    cl::desc("Defaults to TFL Conv ops if channel quantization error is more "
             "than the provided threshold "
             "(default = 0.25)."),
    cl::init(0.25));

cl::opt<bool> convForceErrorCheckOption(
    "xcore-force-conv-err-full-check",
    cl::desc("Enable higher precision(more time-consuming) check for "
             "calculating channel quantization error."),
    cl::init(false));

cl::opt<unsigned> convMultiplierFactorOption(
    "xcore-conv-multiplier-factor",
    cl::desc("If the dynamic range for multipliers is too large, quantization "
             "error increases. This option is a temporary solution to set all "
             "the multipliers to be clamped to a specified multiple of the "
             "minimum multiplier."
             "(default = UINT32_MAX)."),
    cl::init(UINT32_MAX));

} // namespace xcore
} // namespace mlir

LogicalResult runPassPipeline(const PassPipelineCLParser &passPipeline,
                              const OwningOpRef<ModuleOp> &mod,
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
  static cl::opt<std::string> tflmcPrefixOption(
      "xcore-naming-prefix",
      cl::desc("Specify naming prefix(also \"--xp\") for compiled model"
               "(default = \"model_\")."),
      cl::init("model_"));
  static cl::alias aliasTflmcPrefixOption(
      "xp", cl::desc("Alias for --xcore-naming-prefix"),
      cl::aliasopt(tflmcPrefixOption));

  // Register any command line options.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  xcore::registerXCorePassPipeline();
  PassPipelineCLParser passPipeline("", "Compiler passes to run");
  cl::ParseCommandLineOptions(argc, argv);

  // Initialize dialects.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.loadDialect<arith::ArithmeticDialect>();
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

  if (mlir::xcore::threadCountOption < 1 ||
      mlir::xcore::threadCountOption > 8) {
    return failedMessage(
        "Please specify a thread count between one and eight!");
  }

  // Parse input.
  OwningOpRef<ModuleOp> mod;
  SourceMgr sourceMgr;
  if (mlirIOEnabled) {
    // Parse the MLIR input file.
    std::string errorMessage;
    auto file = openInputFile(inputFilename, &errorMessage);
    if (!file) {
      return failedMessage(errorMessage);
    }
    sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
    mod = parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  } else {
    // Read flatbuffer and convert to serialized MLIR string.
    mod = xcore::utils::readFlatBufferFileToMLIR(inputFilename, &context);
    if (!mod) {
      return failedMessage("Unable to read flatbuffer file!");
    }
  }

  // Disable printing op on diagnostics such as error, remark, warning
  context.printOpOnDiagnostic(false);
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

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
    // Translate MLIR to flatbuffer string
    // Prepare metadata
    auto module = mod.get();
    int requiredThreadCount = 1;
    if (auto attr = module->getAttr(xcRequiredThreadCountAttrName)) {
      requiredThreadCount = attr.cast<mlir::IntegerAttr>().getInt();
    }

    struct shared_config::xcore_metadata cfg;
    // Store version info
    cfg.lib_nn_major_version = lib_nn::major_version;
    cfg.lib_nn_minor_version = lib_nn::minor_version;
    cfg.lib_nn_patch_version = lib_nn::patch_version;
    cfg.lib_tflite_micro_major_version = lib_tflite_micro::major_version;
    cfg.lib_tflite_micro_minor_version = lib_tflite_micro::minor_version;
    cfg.lib_tflite_micro_patch_version = lib_tflite_micro::patch_version;
    cfg.xformer_major_version = xformer::majorVersion;
    cfg.xformer_minor_version = xformer::minorVersion;
    cfg.xformer_patch_version = xformer::patchVersion;
    // Store number of threads needed to execute the model
    cfg.required_thread_count = requiredThreadCount;
    auto bufferData =
        std::string((char *)&cfg, sizeof(shared_config::xcore_metadata));

    std::map<std::string, std::string> metadata;
    auto xcoreConfigMetadata =
        std::make_pair(shared_config::xcoreMetadataName, bufferData);
    metadata.insert(xcoreConfigMetadata);

    std::string flatBufferString;
    if (failed(xcore::utils::getFlatBufferStringFromMLIR(
            module, metadata, dontMinifyEnabled, flatBufferString)))
      return failedMessage("Failed to obtain flatbuffer string from MLIR!");

    // Invoke tflmc and get info
    std::stringstream tflmcSourceString, tflmcHeaderString;
    try {
      tflmc::Compiler compiler(flatBufferString.data(), tflmcPrefixOption);
      emitRemark(UnknownLoc::get(module.getContext()))
          << "Tensor arena size : " << compiler.getTensorArenaSize();
      compiler.writeSource(tflmcSourceString);
      compiler.writeHeader(tflmcHeaderString);
    } catch (const std::exception &e) {
      return failedMessage(e.what());
    } catch (...) {
      return failedMessage("Unknown exception while invoking tflmc!");
    }

    std::string outFilename(outputFilename);
    if (failed(xcore::utils::writeDataToFile(outFilename, flatBufferString))) {
      return failedMessage("Failed to write output tflite file!");
    }

    std::string tflmcSourceFilename(outputFilename + ".cpp");
    if (failed(xcore::utils::writeDataToFile(tflmcSourceFilename,
                                             tflmcSourceString.str()))) {
      return failedMessage("Failed to write output source file!");
    }

    std::string tflmcHeaderFilename(outputFilename + ".h");
    if (failed(xcore::utils::writeDataToFile(tflmcHeaderFilename,
                                             tflmcHeaderString.str()))) {
      return failedMessage("Failed to write output header file!");
    }
  }

  return 0;
}
