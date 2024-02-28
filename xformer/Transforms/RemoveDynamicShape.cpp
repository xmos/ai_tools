// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir::xcore {

namespace {
struct RemoveDynamicShape
    : public PassWrapper<RemoveDynamicShape, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveDynamicShape)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<XCoreDialect>();
  }
  StringRef getArgument() const final { return "xcore-remove-dynamic-shape"; }
  StringRef getDescription() const final { return "Remove dynamic shape"; }
  void runOnOperation() override;
};

void RemoveDynamicShape::runOnOperation() {
  auto func = getOperation();
  auto *ctx = &getContext();

  // Lambda for getting a new type with dynamic changed to static
  auto getNewTensorType = [&](TensorType tensorType) {
    TensorType newType = tensorType;
    // If batch dim is dynamic, make it of size one
    if (tensorType.hasRank() && tensorType.getRank() > 1 &&
        tensorType.getDimSize(0) == ShapedType::kDynamic) {
      llvm::ArrayRef<int64_t> shape = tensorType.getShape();
      std::vector<int64_t> newShape;
      newShape.reserve(shape.size());
      for (auto &dim : shape) {
        newShape.push_back(static_cast<int>(dim));
      }
      newShape[0] = 1;
      newType = tensorType.clone(llvm::ArrayRef<int64_t>(newShape));
    }
    return newType;
  };

  // Handle func arguments and return types
  llvm::SmallVector<Type> newFuncInputTypes;
  newFuncInputTypes.resize(func.getNumArguments());
  llvm::SmallVector<Type> newFuncOutputTypes;
  newFuncOutputTypes.resize(func.getNumResults());

  for (BlockArgument argument : func.getArguments()) {
    auto tensorType = argument.getType().dyn_cast<TensorType>();
    auto newType = getNewTensorType(tensorType);
    newFuncInputTypes[argument.getArgNumber()] = newType;
    argument.setType(newType);
  }

  for (int i = 0; i < func.getNumResults(); ++i) {
    auto tensorType = func.getResultTypes()[i].dyn_cast<TensorType>();
    newFuncOutputTypes[i] = getNewTensorType(tensorType);
  }
  FunctionType funcType = func.getFunctionType();
  auto newFuncType =
      FunctionType::get(ctx, newFuncInputTypes, newFuncOutputTypes);
  func.setType(newFuncType);

  // Iterate through all other ops
  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == func) {
      return;
    }

    for (Value result : op->getResults()) {
      if (result.getType().isa<NoneType>()) {
        continue;
      }
      auto tensorType = result.getType().dyn_cast<TensorType>();
      result.setType(getNewTensorType(tensorType));
    }
  });
}
} // namespace

// Creates an instance of the RemoveDynamicShape pass.
std::unique_ptr<OperationPass<func::FuncOp>> createRemoveDynamicShapePass() {
  return std::make_unique<RemoveDynamicShape>();
}

static PassRegistration<RemoveDynamicShape> pass;

} // namespace mlir::xcore
