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
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
// TODO: dpk
// refactor tflmc to have include folder
#include "src/Compiler.h"

using namespace llvm;
using namespace mlir;

namespace mlir::xcore {

// Mark all our options with this category, everything else (except for -version
// and -help) will be hidden.
static cl::OptionCategory XformerCategory("Xformer options");

cl::opt<unsigned>
    quadraticLookupErrorOption("xcore-quadratic-lookup-error",
                               cl::desc("Used only for int16. Defaults to TFL "
                                        "ops if quadratic lookup error is more "
                                        "than provided "
                                        "(default = 1)."),
                               cl::init(1), cl::cat(XformerCategory));

cl::opt<bool> enableBetaFloatOption("xcore-enable-beta-float",
                                    cl::desc("Enable beta float support."),
                                    cl::init(false), cl::cat(XformerCategory));

cl::opt<unsigned> threadCountOption("xcore-thread-count",
                                    cl::desc("[-tc] Thread count"), cl::init(1),
                                    cl::cat(XformerCategory));

cl::alias aliasThreadCountOption("tc",
                                 cl::desc("Alias to --xcore-thread-count"),
                                 cl::aliasopt(threadCountOption));

cl::opt<std::string>
    weightsFilenameOption("xcore-weights-file",
                          cl::desc("[-f] The file to write weights into so "
                                   "that they can be externally loaded."),
                          cl::value_desc("filename"), cl::init(""),
                          cl::cat(XformerCategory));

cl::alias aliasWeightsFilenameOption("f",
                                     cl::desc("Alias for --xcore-weights-file"),
                                     cl::aliasopt(weightsFilenameOption));

cl::opt<bool> tileLoadOption("xcore-load-tile",
                             cl::desc("Enable loading weights from a tile."),
                             cl::init(false), cl::cat(XformerCategory));

cl::opt<unsigned> loadExternallyIfLargerOption(
    "xcore-load-externally-if-larger",
    cl::desc("Load constants externally if larger than given limit in bytes "
             "(default = 96 bytes). Cannot be specified when "
             "xcore-weights-file is not provided."),
    cl::init(96), cl::cat(XformerCategory), cl::Hidden);

cl::opt<unsigned> maxLoadExternalSizeOption(
    "xcore-max-load-external-size",
    cl::desc("The size of external load image from flash or tile will be "
             "limited to the max specified bytes "
             "(default = UINT_MAX bytes)."),
    cl::init(UINT_MAX), cl::cat(XformerCategory), cl::Hidden);

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
    cl::init(0.25), cl::cat(XformerCategory));

cl::opt<bool> convForceErrorCheckOption(
    "xcore-force-conv-err-full-check",
    cl::desc("Enable higher precision(more time-consuming) check for "
             "calculating channel quantization error."),
    cl::init(false), cl::cat(XformerCategory), cl::Hidden);

cl::opt<unsigned> convMultiplierFactorOption(
    "xcore-conv-multiplier-factor",
    cl::desc("If the dynamic range for multipliers is too large, quantization "
             "error increases. This option is a temporary solution to set all "
             "the multipliers to be clamped to a specified multiple of the "
             "minimum multiplier."
             "(default = UINT32_MAX)."),
    cl::init(UINT32_MAX), cl::cat(XformerCategory), cl::Hidden);

cl::opt<bool> opSplitTensorArenaOption(
    "xcore-op-split-tensor-arena",
    cl::desc("Enable prototype op split to reduce tensor arena size."),
    cl::init(false), cl::cat(XformerCategory));

cl::opt<unsigned>
    opSplitTargetSizeOption("xcore-op-split-target-size",
                            cl::desc("Op split target max tensor arena size."),
                            cl::init(700000), cl::cat(XformerCategory));

cl::list<unsigned>
    opSplitBottomOpsOption("xcore-op-split-bottom-op",
                           cl::desc("Manual override Op split, bottom op."),
                           cl::CommaSeparated, cl::cat(XformerCategory));

cl::list<unsigned>
    opSplitTopOpsOption("xcore-op-split-top-op",
                        cl::desc("Manual override Op split, top op."),
                        cl::CommaSeparated, cl::cat(XformerCategory));

cl::list<unsigned> opSplitNumSplitsOption(
    "xcore-op-split-num-splits",
    cl::desc("Manual override Op split, number of splits."), cl::CommaSeparated,
    cl::cat(XformerCategory));

cl::opt<bool> allowInputModificationOption(
    "xcore-allow-input-modification",
    cl::desc("Allow the compiler to modify input tensor for optimizations."),
    cl::init(false), cl::cat(XformerCategory), cl::Hidden);

cl::opt<bool> convDebugOption("xcore-conv-debug",
                              cl::desc("Enable conv debug prints."),
                              cl::init(false), cl::cat(XformerCategory),
                              cl::Hidden);

cl::opt<bool> overlapConvOption("xcore-overlap-conv",
                                cl::desc("Overlap conv also."), cl::init(false),
                                cl::cat(XformerCategory), cl::Hidden);

cl::opt<bool> offlineOffsetsOption("xcore-offline-offsets",
                                   cl::desc("Offline offsets"), cl::init(true),
                                   cl::cat(XformerCategory), cl::Hidden);

cl::opt<unsigned> convChannelwiseSplitSizeOption(
    "xcore-conv-channelwise-split-size",
    cl::desc(
        "Specify channelwise split size for convolutions (default = 100000)."),
    cl::init(100000), cl::cat(XformerCategory), cl::Hidden);

} // namespace mlir::xcore

static LogicalResult runPassPipeline(const PassPipelineCLParser &passPipeline,
                                     const OwningOpRef<ModuleOp> &mod,
                                     MLIRContext *context) {
  auto module = mod.get();
  PassManager pm(module->getName(), mlir::OpPassManager::Nesting::Implicit);
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

static LogicalResult isCompatibleVersion(cl::opt<std::string> &version,
                                         int32_t majorVersion,
                                         int32_t minorVersion,
                                         int32_t patchVersion) {
  if (!version.empty()) {
    SmallVector<StringRef> partsStr;
    llvm::SplitString(version, partsStr, ".");
    if (partsStr.size() != 3) {
      return failure();
    }
    SmallVector<int> parts;
    int val = 0;
    for (auto &i : partsStr) {
      if (!llvm::to_integer(i, val, 10)) {
        return failure();
      }
      parts.push_back(val);
    }

    // Check provided repo version with compiler version
    // If major version is zero, then minor versions must match
    // Otherwise, major versions must match and compiler version
    // must be less or equal to provided repo version
    if ((majorVersion == 0 && parts[0] == 0 && minorVersion != parts[1]) ||
        (majorVersion != parts[0]) || (minorVersion > parts[1])) {
      return failure();
    }
  }
  return success();
}

static void PrintVersion(raw_ostream &OS) {
  OS << xformer::majorVersion << "." << xformer::minorVersion << "."
     << xformer::patchVersion << '\n';
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Override the default '-h' and use the default PrintHelpMessage()
  static llvm::cl::opt<bool> help("h", llvm::cl::desc("Alias for -help"),
                                  llvm::cl::Hidden);

  static cl::opt<std::string> inputFilename(cl::Positional,
                                            cl::desc("<TFLite FlatBuffer>"));
  static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"));
  static cl::opt<bool> mlirIOEnabled(
      "mlir-io", cl::desc("Enable MLIR input and output"), cl::init(false),
      cl::cat(mlir::xcore::XformerCategory), cl::Hidden);
  static cl::opt<bool> verifyDiagnosticsEnabled(
      "verify-diagnostics",
      cl::desc("Check that emitted diagnostics match "
               "expected-* lines on the corresponding line"),
      cl::init(false));
  static cl::opt<bool> dontMinifyEnabled(
      "xcore-dont-minify",
      cl::desc("Do not strip debug info and minify the model"), cl::init(false),
      cl::cat(mlir::xcore::XformerCategory), cl::Hidden);
  static cl::opt<std::string> tflmcPrefixOption(
      "xcore-naming-prefix",
      cl::desc("[-xp] Specify naming prefix for compiled model"
               "(default = \"model_\")."),
      cl::init("model_"), cl::cat(mlir::xcore::XformerCategory));
  static cl::alias aliasTflmcPrefixOption(
      "xp", cl::desc("Alias for --xcore-naming-prefix"),
      cl::aliasopt(tflmcPrefixOption), cl::cat(mlir::xcore::XformerCategory));
  static cl::opt<bool> tflmcPrintEnabled(
      "xcore-tflmc-print", cl::desc("Print out memory allocation plan"),
      cl::init(false), cl::cat(mlir::xcore::XformerCategory));
  static cl::opt<std::string> versionLibTfliteMicro(
      "xcore-compatible-with-lib-tflite-micro",
      cl::desc("Check if lib_tflite_micro version is compatible"), cl::init(""),
      cl::cat(mlir::xcore::XformerCategory), cl::Hidden);
  static cl::opt<std::string> versionLibNN(
      "xcore-compatible-with-lib-nn",
      cl::desc("Check if lib_nn version is compatible"), cl::init(""),
      cl::cat(mlir::xcore::XformerCategory), cl::Hidden);

  // Register any command line options.
  registerPassManagerCLOptions();
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  xcore::registerXCorePassPipeline();
  PassPipelineCLParser passPipeline("", "Compiler passes to run");
  cl::SetVersionPrinter(PrintVersion);
  cl::HideUnrelatedOptions(mlir::xcore::XformerCategory);
  cl::ParseCommandLineOptions(argc, argv);
  if (help) {
    llvm::cl::PrintHelpMessage();
    return 0;
  }

  // Initialize dialects.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.loadDialect<arith::ArithDialect>();
  context.loadDialect<quant::QuantizationDialect>();
  context.loadDialect<TFL::TensorFlowLiteDialect>();
  context.loadDialect<xcore::XCoreDialect>();
  context.printOpOnDiagnostic(!verifyDiagnosticsEnabled);

  auto failedMessage = [&](const Twine &msg) {
    emitError(UnknownLoc::get(&context)) << msg;
    return 1;
  };

  // Validate options
  if (mlir::xcore::tileLoadOption.getNumOccurrences() > 0 &&
      mlir::xcore::threadCountOption < 4) {
    return failedMessage("Please specify at least four threads using "
                         "xcore-thread-count option when using the "
                         "xcore-load-tile option!");
  }

  if (mlir::xcore::loadExternallyIfLargerOption.getNumOccurrences() > 0 &&
      mlir::xcore::weightsFilenameOption.empty()) {
    return failedMessage(
        "Please specify the xcore-weights-file option when specifying the "
        "xcore-load-externally-if-larger option!");
  }

  if (mlir::xcore::opSplitTargetSizeOption.getNumOccurrences() > 0 &&
      (!(mlir::xcore::opSplitBottomOpsOption.empty()) ||
       !(mlir::xcore::opSplitTopOpsOption.empty()) ||
       !(mlir::xcore::opSplitNumSplitsOption.empty()))) {
    return failedMessage(
        "Target size option cannot be used with start, end, and "
        "numSplits options");
  }

  if (mlir::xcore::threadCountOption < 1 ||
      mlir::xcore::threadCountOption > 5) {
    return failedMessage("Please specify a thread count between one and five!");
  }

  if (failed(isCompatibleVersion(
          versionLibTfliteMicro, lib_tflite_micro::major_version,
          lib_tflite_micro::minor_version, lib_tflite_micro::patch_version))) {
    return failedMessage("Incompatible lib_tflite_micro version!\n\nPlease use "
                         "lib_tflite_micro version " +
                         Twine(lib_tflite_micro::major_version) + "." +
                         Twine(lib_tflite_micro::minor_version) + "." +
                         Twine(lib_tflite_micro::patch_version));
  }

  if (failed(isCompatibleVersion(versionLibNN, lib_nn::major_version,
                                 lib_nn::minor_version,
                                 lib_nn::patch_version))) {
    return failedMessage("Incompatible lib_nn version!\n\nPlease use "
                         "lib_nn version " +
                         Twine(lib_nn::major_version) + "." +
                         Twine(lib_nn::minor_version) + "." +
                         Twine(lib_nn::patch_version));
  }

  // Parse input.
  OwningOpRef<ModuleOp> mod;
  SourceMgr sourceMgr;
  if (mlirIOEnabled) {
    // Parse the MLIR input file.
    std::string errorMessage;
    auto file = mlir::openInputFile(inputFilename, &errorMessage);
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
    auto output = mlir::openOutputFile("-", &errorMessage);
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

    struct shared_config::xcore_metadata sharedCfg;
    // Store version info
    sharedCfg.lib_nn_major_version = lib_nn::major_version;
    sharedCfg.lib_nn_minor_version = lib_nn::minor_version;
    sharedCfg.lib_nn_patch_version = lib_nn::patch_version;
    sharedCfg.lib_tflite_micro_major_version = lib_tflite_micro::major_version;
    sharedCfg.lib_tflite_micro_minor_version = lib_tflite_micro::minor_version;
    sharedCfg.lib_tflite_micro_patch_version = lib_tflite_micro::patch_version;
    sharedCfg.xformer_major_version = xformer::majorVersion;
    sharedCfg.xformer_minor_version = xformer::minorVersion;
    sharedCfg.xformer_patch_version = xformer::patchVersion;
    // Store number of threads needed to execute the model
    sharedCfg.required_thread_count = mlir::xcore::threadCountOption;
    auto bufferData =
        std::string((char *)&sharedCfg, sizeof(shared_config::xcore_metadata));

    std::map<std::string, std::string> metadata;
    auto xcoreConfigMetadata =
        std::make_pair(shared_config::xcoreMetadataName, bufferData);

    // Offline offsets metadata

    // std::vector<int> offline_offsets = {
    //    73728, -1, -1, -1, -1, -1, -1, 0, 129024, 73728, 166272, 132096,
    //    73728, 153984, 132096, 73728, 132096, 73728, 0, 52224, 0};
    if (auto attr = module->getAttr("xc.offsets")) {
      auto offline_offsets = std::vector<int>{
          attr.cast<mlir::DenseIntElementsAttr>().getValues<int32_t>().begin(),
          attr.cast<mlir::DenseIntElementsAttr>().getValues<int32_t>().end()};

      constexpr char kOfflineMemAllocMetadata[] = "OfflineMemoryAllocation";
      /*
      | 0 | Offline allocation format version |
      | 1 | Subgraph index to which this allocation applies |
      | 2 | Number offsets following: n |
      | 3 | Byte offset of tensor #0 or -1 to allocate at runtime |
      | 4 | Byte offset of tensor #1 or -1 to allocate at runtime |
      | ... | ... |
      | 3+(n-1) | Byte offset of tensor #(n-1) or -1 to allocate at runtime |
      */
      offline_offsets.insert(offline_offsets.begin(),
                             {0, 0, (int)offline_offsets.size()});
      // Align to sixteen bytes as metadata value has to be 16-byte aligned
      // buffer
      offline_offsets.resize(((offline_offsets.size() + 3) / 4) * 4);

      auto offlineOffsetsData = std::string((char *)offline_offsets.data(),
                                            offline_offsets.size() * 4);

      auto k = (int32_t *)offlineOffsetsData.data();
#define DEBUG_TYPE "xcore-memory-plan"
      LLVM_DEBUG(llvm::dbgs() << "\n\n");
      for (int i = 0; i < offline_offsets.size(); i++) {
        LLVM_DEBUG(llvm::dbgs() << k[i] << ", ");
      }
      LLVM_DEBUG(llvm::dbgs() << "\n\n");

      auto offlineOffsetsMetadata =
          std::make_pair(kOfflineMemAllocMetadata, offlineOffsetsData);

      LLVM_DEBUG(llvm::dbgs() << "\n\nOFFLINE OFFSETS ENABLED!\n\n");
#undef DEBUG_TYPE

      metadata.insert(offlineOffsetsMetadata);
    }
    metadata.insert(xcoreConfigMetadata);

    std::string flatBufferString;
    if (failed(xcore::utils::getFlatBufferStringFromMLIR(
            module, metadata, dontMinifyEnabled, flatBufferString))) {
      return failedMessage("Failed to obtain flatbuffer string from MLIR!");
    }

    // Write tflite file
    std::string outFilename(outputFilename);
    if (failed(xcore::utils::writeDataToFile(outFilename, flatBufferString))) {
      return failedMessage("Failed to write output tflite file!");
    }

    // Invoke tflmc and get info
    std::stringstream tflmcSourceString, tflmcHeaderString;
    try {
      tflmc::Compiler compiler(flatBufferString.data(), &sharedCfg,
                               tflmcPrefixOption, tflmcPrintEnabled);
      emitRemark(UnknownLoc::get(module.getContext()))
          << "Tensor arena size : " << compiler.getTensorArenaSize();
      compiler.writeSource(tflmcSourceString);
      compiler.writeHeader(tflmcHeaderString);
    } catch (const std::exception &e) {
      return failedMessage(e.what());
    } catch (...) {
      return failedMessage("Unknown exception while invoking tflmc!");
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
