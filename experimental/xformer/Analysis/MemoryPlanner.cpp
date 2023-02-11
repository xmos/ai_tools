// Copyright 2023 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Analysis/MemoryPlanner.h"

#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {

MemoryPlanner::MemoryPlanner(func::FuncOp op) : funcOp(op), liveness(op) {
  auto getValueSize = [](Value v) {
    auto type = v.getType().dyn_cast<ShapedType>();
    size_t typeSizeInBytes;
    if (type.getElementType().isa<quant::QuantizedType>()) {
      // we only support QI8
      typeSizeInBytes = 1;
    } else {
      typeSizeInBytes =
          type.getElementType().getIntOrFloatBitWidth() / CHAR_BIT;
    }

    size_t k = typeSizeInBytes;
    llvm::ArrayRef<int64_t> shape_ref = type.getShape();
    for (auto &dim : shape_ref) {
      k *= (dim == -1 ? 1 : dim);
    }

    // Align size up to four bytes
    k = ((k + 3) / 4) * 4;

    return k;
  };

  for (BlockArgument argument : funcOp.getArguments()) {
    valueInfo.insert(
        {argument, {valueInfo.size(), getValueSize(argument), false, -1, -1}});
    values.push_back(argument);
  }

  funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == funcOp) {
      return;
    }

    operationIds.insert({op, operationIds.size()});
    operations.push_back(op);

    bool isConstantOp = false;
    // TODO(renjieliu): Find a generic way to deal with const ops.
    if (op->hasTrait<OpTrait::IsTerminator>() ||
        llvm::isa<TFL::QConstOp, TFL::ConstOp, arith::ConstantOp>(op)) {
      isConstantOp = true;
    }

    for (Value result : op->getResults()) {
      valueInfo.insert(
          {result,
           {valueInfo.size(), getValueSize(result), isConstantOp, -1, -1}});
      values.push_back(result);
    }
  });

  //   for (auto v : values) {
  //   auto type = v.getType().dyn_cast<ShapedType>();
  //   int k = 1;
  //   llvm::ArrayRef<int64_t> shape_ref = type.getShape();
  //   for (auto &dim : shape_ref) {
  //     k *= (dim == -1 ? 1 : dim);
  //   }
  //   valueSizes[v] = k;
  // }

  // Liveness
  // Struct with start op and end op
  assert(op->getNumRegions() == 1);
  assert(op->getRegion(0).hasOneBlock());

  Block *block = &op->getRegion(0).front();

  const LivenessBlockInfo *lvb = liveness.getLiveness(block);
  for (auto v : values) {
    Operation *startOp = lvb->getStartOperation(v);
    valueInfo[v].firstUsed = operationIds[startOp];
    valueInfo[v].lastUsed = operationIds[lvb->getEndOperation(v, startOp)];
  }

  printf("\n\n");

  for (auto v : values) {
    printf("\nvalue %d size = %d start op = %d end op = %d", valueInfo[v].id,
           valueInfo[v].size, valueInfo[v].firstUsed, valueInfo[v].lastUsed);
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
      if (!valueInfo[v].constant)
        size += valueInfo[v].size;
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

  if (valueInfo[v].constant) {
    return -1;
  }

  // Go through all selected buffers
  // They are ordered by offset

  for (auto i : selected) {
    Value c = i.first;
    int cOffset = i.second;

    if ((valueInfo[c].firstUsed > valueInfo[v].lastUsed) ||
        (valueInfo[v].firstUsed > valueInfo[c].lastUsed)) {
      // no overlap
      continue;
    }

    // overlapping buffer
    if (cOffset - possibleOffset > size) {
      // there is a gap
      break;
    } else {
      // move offset to end of current buffer if larger
      int end = cOffset + valueInfo[c].size;

      if (end > possibleOffset) {
        possibleOffset = end;
      }
    }
  }

  return possibleOffset;
}

std::vector<int> MemoryPlanner::getOffsets() {
  std::vector<int> offsets;

  auto OrderedDescendingSizesComparator = [&](QueueItem &lhs, QueueItem &rhs) {
    if (lhs.second != rhs.second) {
      return lhs.second < rhs.second;
    }
    return valueInfo[lhs.first].id < valueInfo[rhs.first].id;
  };
  // The top item is the largest one.
  llvm::PriorityQueue<QueueItem, std::vector<QueueItem>,
                      decltype(OrderedDescendingSizesComparator)>
      queue(OrderedDescendingSizesComparator);
  // insert all values and size into priority queue
  for (auto v : values) {
    queue.push({v, valueInfo[v].size});
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
    return valueInfo[a.first].id < valueInfo[b.first].id;
  };
  std::multiset<QueueItem, decltype(cmp)> Offsets(cmp);
  for (auto i : selected) {
    Offsets.insert(i);
  }

  printf("\nAllocated offsets : ");
  for (auto i : Offsets) {
    offsets.push_back(i.second);
    printf("\nValue %d, size %d, offset %d ", valueInfo[i.first].id,
           valueInfo[i.first].size, i.second);
  }
  printf("\n\n");

  // funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
  //   if (op == funcOp) {
  //     return;
  //   }

  //   // TODO(renjieliu): Find a generic way to deal with const ops.
  //   if (op->hasTrait<OpTrait::IsTerminator>() ||
  //       llvm::isa<TFL::QConstOp, TFL::ConstOp, arith::ConstantOp>(op)) {
  //     for (Value result : op->getResults()) {
  //       offsets.push_back(-1);
  //     }
  //   } else {
  //     for (Value result : op->getResults()) {
  //       if (auto search = Offsets.find(result); search != Offsets.end()) {
  //         offsets.push_back(search->second);
  //       } else {
  //         assert(false);
  //       }
  //     }
  //   }
  // });

  // printf("\n\n");

  // for (auto i : offsets) {
  //   printf("%d, ", i);
  // }
  // printf("\n\n");

  return offsets;
}

} // namespace xcore
} // namespace mlir
