// Copyright 2023 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Analysis/MemoryPlanner.h"

#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {

MemoryPlanner::MemoryPlanner(func::FuncOp op) : funcOp(op), liveness(op) {
  for (BlockArgument argument : funcOp.getArguments()) {
    valueIds.insert({argument, valueIds.size()});
    values.push_back(argument);
  }

  funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
    operationIds.insert({op, operationIds.size()});
    operations.push_back(op);

    // TODO(renjieliu): Find a generic way to deal with const ops.
    if (op->hasTrait<OpTrait::IsTerminator>() ||
        llvm::isa<TFL::QConstOp, TFL::ConstOp, arith::ConstantOp>(op))
      return;

    for (Value result : op->getResults()) {
      valueIds.insert({result, valueIds.size()});
      values.push_back(result);
    }
  });
}

int MemoryPlanner::getNewOffset(Value v, int size,
                                std::vector<std::pair<Value, int>> &selected) {
  int newOffset = 0;

  //Liveness liveness(func);
  // for (auto &i : valueIds) {
  //   llvm::errs() << i.second << " ";
  //   i.first.dump();
  //   llvm::errs() << "\n";

  //   Liveness::OperationListT k = liveness.resolveLiveness(i.first);
  //   for(auto &o : k){
  //     llvm::errs() << operationIds[o] << " ";
  //   }
  //   llvm::errs() << "\n\n";
  // }

  return newOffset;
}

std::vector<int> MemoryPlanner::getOffsets() {
  std::vector<int> offsets;
  // insert all values and size into priority queue
  for (auto v : values) {
    auto type = v.getType().dyn_cast<ShapedType>();

    int k;
    if (type.getElementType().isa<quant::QuantizedType>()) {
      if (type.getElementType().cast<quant::QuantizedType>().isSigned() &&
          type.getElementType()
                  .cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == 8) {
        k = type.getNumElements();
      } else {
        assert(false);
      }
    } else {
      k = type.getSizeInBits() / 8;
    }

    queue.push({v, k});
  }

  std::vector<std::pair<Value, int>> selected;

  // Add first to selected list with offset zero
  // while priority queue is not empty()
  // pop and check with selected list

  auto v = queue.top().first;
  queue.pop();
  selected.push_back({v, 0});

  printf("\n");
  while (!queue.empty()) {
    auto v = queue.top().first;
    auto size = queue.top().second;
    printf("%d ", size);
    queue.pop();

    // check with selected list
    int newOffset = getNewOffset(v, size, selected);
    selected.push_back({v, newOffset});
  }
  printf("\n");
  return offsets;
}

} // namespace xcore
} // namespace mlir
