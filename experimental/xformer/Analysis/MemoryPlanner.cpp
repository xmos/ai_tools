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
    if (op == funcOp) {
      return;
    }

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

  // Liveness
  // Struct with start op and end op
  assert(op->getNumRegions() == 1);
  assert(op->getRegion(0).hasOneBlock());

  Block *block = &op->getRegion(0).front();

  const LivenessBlockInfo *lvb = liveness.getLiveness(block);
  for (auto v : values) {
    Operation *startOp = lvb->getStartOperation(v);
    livenessInfo[v].firstUsed = operationIds[startOp];
    livenessInfo[v].lastUsed = operationIds[lvb->getEndOperation(v, startOp)];
  }

  for (auto v : values) {
    auto type = v.getType().dyn_cast<ShapedType>();
    int k = 1;
    llvm::ArrayRef<int64_t> shape_ref = type.getShape();
    for (auto &dim : shape_ref) {
      k *= (dim == -1 ? 1 : dim);
    }

    // if (type.getElementType().isa<quant::QuantizedType>()) {
    //   if (type.getElementType().cast<quant::QuantizedType>().isSigned() &&
    //       type.getElementType()
    //               .cast<quant::QuantizedType>()
    //               .getStorageTypeIntegralWidth() == 8) {
    //     k = type.getNumElements();
    //   } else {
    //     // Not QI8
    //     assert(false);
    //   }
    // } else {
    //   k = type.getSizeInBits() / 8;
    // }

    valueSizes[v] = k;
  }
  printf("\n\n");

  for (auto v : values) {
    printf("\nvalue %d size = %d start op = %d end op = %d", valueIds[v],
           valueSizes[v], livenessInfo[v].firstUsed, livenessInfo[v].lastUsed);
  }
  printf("\n\n");

  int maxSize = -1;
  Operation *maxOp;
  for (auto o : operations) {
    if (o->hasTrait<OpTrait::IsTerminator>() ||
        llvm::isa<TFL::QConstOp, TFL::ConstOp, arith::ConstantOp>(o)) {
      continue;
    }
    int size = 0;
    for (auto v : lvb->currentlyLiveValues(o)) {
      size += valueSizes[v];
    }
    if (size > maxSize) {
      maxSize = size;
      maxOp = o;
    }
    printf("\nop %d width = %d", operationIds[o], size);
  }
  printf("\nMax op %d width = %d", operationIds[maxOp], maxSize);
  maxOp->dump();
  maxOp->getLoc().dump();
  printf("\n\n");
}

int MemoryPlanner::getNewOffset(Value v, int size, OrderedOffsets &selected) {
  int possibleOffset = 0;

  // Go through all selected buffers
  // They are ordered by offset

  for (auto i : selected) {
    Value c = i.first;
    int cOffset = i.second;

    if ((livenessInfo[v].firstUsed >= livenessInfo[c].firstUsed &&
         livenessInfo[v].firstUsed <= livenessInfo[c].lastUsed) ||
        (livenessInfo[v].lastUsed >= livenessInfo[c].firstUsed &&
         livenessInfo[v].lastUsed <= livenessInfo[c].lastUsed)) {
      // overlap

      if (cOffset - possibleOffset > size) {
        // there is a gap
        break;
      } else {
        // not enough space
        // move offset to end of current buffer
        int end = cOffset + valueSizes[c];

        if (end > possibleOffset) {
          possibleOffset = end;
        }
      }
    }
  }

  return possibleOffset;
}

std::vector<int> MemoryPlanner::getOffsets() {
  std::vector<int> offsets;
  // insert all values and size into priority queue
  for (auto v : values) {
    queue.push({v, valueSizes[v]});
  }

  // ordered by offset
  OrderedOffsets selected;
  Value l;

  // Add first to selected list with offset zero
  // while priority queue is not empty()
  // pop and check with selected list

  auto v = queue.top().first;
  queue.pop();
  selected.insert({v, 0});

  printf("\nSorted buffers : ");
  while (!queue.empty()) {
    auto v = queue.top().first;
    auto size = queue.top().second;
    printf("%d ", size);
    queue.pop();

    // check with selected list
    int newOffset = getNewOffset(v, size, selected);
    selected.insert({v, newOffset});
  }
  printf("\n\n");

  auto cmp = [&](QueueItem a, QueueItem b) {
    return valueIds[a.first] < valueIds[b.first];
  };
  std::multiset<QueueItem, decltype(cmp)> Offsets(cmp);
  for (auto i : selected) {
    Offsets.insert(i);
  }

  printf("\nAllocated offsets : ");
  for (auto i : Offsets) {
    printf("\nValue %d, size %d, offset %d ", valueIds[i.first],
           valueSizes[i.first], i.second);
  }
  printf("\n\n");

  return offsets;
}

} // namespace xcore
} // namespace mlir
