// Copyright 2023 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Analysis/MemoryPlan.h"
#include "IR/XCoreOps.h"
#include "Transforms/Options.h"

#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {

MemoryPlan::MemoryPlan(Operation *operation)
    : liveness(operation), op(operation) {
  build();
}

void MemoryPlan::build() {
  if (!llvm::isa<func::FuncOp>(op)) {
    return;
  }

  auto funcOp = dyn_cast<func::FuncOp>(op);

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
}

int MemoryPlan::getMaxMemoryUsed() {
  Block *block = &op->getRegion(0).front();
  const LivenessBlockInfo *lvb = liveness.getLiveness(block);

  int maxSize = -1;
  Operation *maxOp;
  for (auto o : operations) {
    if (o->hasTrait<OpTrait::IsTerminator>() ||
        llvm::isa<TFL::QConstOp, TFL::ConstOp, arith::ConstantOp>(o)) {
      continue;
    }
    int size = 0;
    for (auto v : lvb->currentlyLiveValues(o)) {
      if (!valueInfo[v].isConstant)
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
  return maxSize;
}

int MemoryPlan::getOffset(Value v, int size,
                          ValuesOrderedByOffset &allocatedValues) {
  int offset = 0;

  // Go through all allocated buffers
  // They are ordered by offset
  for (auto i : allocatedValues) {
    Value allocatedVal = i.first;
    int allocatedOffset = i.second;

    if ((valueInfo[allocatedVal].firstUsed > valueInfo[v].lastUsed) ||
        (valueInfo[v].firstUsed > valueInfo[allocatedVal].lastUsed)) {
      // No overlap
      continue;
    }

    // Found an overlapping buffer
    if (allocatedOffset - offset > size) {
      // There is a gap
      break;
    } else {
      // Move offset to end of current buffer if larger
      int end = allocatedOffset + valueInfo[allocatedVal].size;
      if (end > offset) {
        offset = end;
      }
    }
  }

  return offset;
}

std::vector<int> MemoryPlan::getAllocatedOffsets() {
  std::vector<int> offsets;

  // Find linked values,
  // increase input value size
  // add only input values,
  // stitch up after allocation and fix input and output value offsets
  llvm::DenseMap<Value, std::pair<Value, int>> outInVals;
  if (overlapOption) {
    for (auto o : operations) {
      if (llvm::isa<PadOp>(o)) {
        auto in = o->getOperand(0);
        if (in.hasOneUse()) {
          auto out = o->getResult(0);
          int offset = valueInfo[out].size - valueInfo[in].size;
          outInVals[out] = {in, offset};
          valueInfo[in].size += offset;
          valueInfo[in].lastUsed = valueInfo[out].lastUsed;
        }
      }

      // if (llvm::isa<Conv2DV2Op>(o)) {
      //   auto convOp = dyn_cast<Conv2DV2Op>(o);
      //   if (symbolizeConv2DType(convOp.conv2d_kernel_type()) !=
      //       Conv2DType::ValidIndirect) {
      //     continue;
      //   }
      //   auto in = o->getOperand(0);
      //   auto out = o->getResult(0);
      //   int offset = 576;//valueInfo[out].size - valueInfo[in].size;
      //   outInVals[out] = {in, offset};
      //   valueInfo[in].size += offset;
      //   valueInfo[in].lastUsed = valueInfo[out].lastUsed;
      // }
    }
  }

  // Fix up consecutive overlapping allocations
  // for (auto val : inputOutputPair) {
  //   Value currentInput = val.first;
  //   while(inputOutputPair.count(currentInput)) {
  //     currentInput = inputOutputPair[currentInput];
  //   }
  //   inputOutputPair[val.first] = inputOutputPair[currentInput];
  // }

  // The comparator keeps the buffers ordered by id if their sizes are the same
  auto DecreasingSizesComparator = [&](QueueItem &lhs, QueueItem &rhs) {
    if (lhs.second != rhs.second) {
      return lhs.second < rhs.second;
    }
    return valueInfo[lhs.first].id < valueInfo[rhs.first].id;
  };
  // The top item is the largest one.
  llvm::PriorityQueue<QueueItem, std::vector<QueueItem>,
                      decltype(DecreasingSizesComparator)>
      queue(DecreasingSizesComparator);

  // Insert values and their sizes into priority queue
  for (auto v : values) {
    if (!outInVals.count(v) && !valueInfo[v].isConstant) {
      queue.push({v, valueInfo[v].size});
    }
  }

  ValuesOrderedByOffset allocatedValues;
  auto v = queue.top().first;
  queue.pop();
  allocatedValues.insert({v, 0});

  printf("\nSorted buffers : ");
  while (!queue.empty()) {
    auto v = queue.top().first;
    auto size = queue.top().second;
    printf("%d ", size);
    queue.pop();

    // check with allocatedValues list
    int newOffset = getOffset(v, size, allocatedValues);
    allocatedValues.insert({v, newOffset});
  }
  printf("\n\n");

  // Patch up overlapped buffers
  for (auto val : outInVals) {
    auto out = val.first;
    auto in = val.second.first;
    auto offset = val.second.second;

    auto it = std::find_if(allocatedValues.begin(), allocatedValues.end(),
                           [&](const QueueItem &p) { return p.first == in; });

    if (it != allocatedValues.end()) {
      int currentOffset = it->second;
      allocatedValues.erase(it);
      allocatedValues.insert({in, currentOffset + offset});
      allocatedValues.insert({out, currentOffset});
    } else {
      assert(false);
    }
  }

  for (auto v : values) {
    if (valueInfo[v].isConstant) {
      allocatedValues.insert({v, -1});
    }
  }

  // Sort the allocated offsets by id, i.e., execution order
  auto cmp = [&](QueueItem a, QueueItem b) {
    return valueInfo[a.first].id < valueInfo[b.first].id;
  };
  std::multiset<QueueItem, decltype(cmp)> allocatedValuesOrderedByID(cmp);
  for (auto i : allocatedValues) {
    allocatedValuesOrderedByID.insert(i);
  }

  printf("\nAllocated offsets : ");
  for (auto i : allocatedValuesOrderedByID) {
    offsets.push_back(i.second);
    printf("\nValue %d, size %d, offset %d ", valueInfo[i.first].id,
           valueInfo[i.first].size, i.second);
  }
  printf("\n\n");

  return offsets;
}

} // namespace xcore
} // namespace mlir
