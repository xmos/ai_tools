// Copyright 2023 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Analysis/MemoryPlan.h"
#include "IR/XCoreOps.h"
#include "Transforms/Options.h"
#include "Utils/Util.h"

#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "xcore-memory-plan"

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

  auto getAlignedValueSize = [](Value v) {
    auto type = v.getType().dyn_cast<ShapedType>();
    size_t k = static_cast<size_t>(utils::getShapedTypeSize(type));
    // Align size up to double word = 8 bytes
    k = ((k + 7) / 8) * 8;
    return k;
  };

  for (BlockArgument argument : funcOp.getArguments()) {
    valueInfo.insert(
        {argument,
         {valueInfo.size(), getAlignedValueSize(argument), false, -1, -1}});
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
        llvm::isa<TFL::NoValueOp, TFL::QConstOp, TFL::ConstOp,
                  arith::ConstantOp>(op)) {
      isConstantOp = true;
    }

    for (Value result : op->getResults()) {
      if (result.getType().isa<NoneType>()) {
        continue;
      }
      valueInfo.insert({result,
                        {valueInfo.size(), getAlignedValueSize(result),
                         isConstantOp, -1, -1}});
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
}

Operation *MemoryPlan::getOpWithMaxMemoryUsed() {
  Block *block = &op->getRegion(0).front();
  const LivenessBlockInfo *lvb = liveness.getLiveness(block);

  int maxSize = -1;
  Operation *maxOp;
  for (auto o : operations) {
    if (o->hasTrait<OpTrait::IsTerminator>() ||
        llvm::isa<TFL::NoValueOp, TFL::QConstOp, TFL::ConstOp,
                  arith::ConstantOp>(o)) {
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
    LLVM_DEBUG(llvm::dbgs()
               << "\nop " << operationIds[o] << " width = " << size);
  }
  LLVM_DEBUG(llvm::dbgs() << "\nMax op " << operationIds[maxOp]
                          << " width = " << maxSize);
  LLVM_DEBUG(llvm::dbgs() << "\n\n");
  return maxOp;
}

int MemoryPlan::getOffset(Value v, int size,
                          DenseMap<Value, ValueInfo> &valueInfo,
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
    if (allocatedOffset - offset >= size) {
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

std::vector<int> MemoryPlan::getAllocatedOffsets(const bool overlapOps,
                                                 int &peakMemoryUsed) {
  std::vector<int> offsets;
  // Copy of valueInfo
  auto vInfo = valueInfo;

  // Overlap buffers
  llvm::DenseMap<Value, std::pair<Value, int>> outInVals;
  // outInInVals are only used when overlapping conv and pad together
  llvm::DenseMap<Value, std::pair<std::pair<Value, int>, std::pair<Value, int>>>
      outInInVals;

  int maxOpId = -1;
  if (overlapConvOption) {
    // TODO: Try overlap conv
    // Need to revert conv to run single-threaded which is not implemented yet
    auto maxOp = getOpWithMaxMemoryUsed();
    // max op is usually pad or conv
    // if max op is pad, we choose the next one which should be conv
    if (llvm::isa<Conv2DV2Op>(maxOp)) {
      maxOpId = operationIds[maxOp];
    } else if (llvm::isa<PadOp>(maxOp) &&
               llvm::isa<Conv2DV2Op>(operations[operationIds[maxOp] + 1])) {
      maxOpId = operationIds[maxOp] + 1;
    }
  }

  if (overlapOps) {
    for (auto o : operations) {
      if (llvm::isa<PadOp>(o)) {
        auto in = o->getOperand(0);
        if (in.hasOneUse()) {
          auto out = o->getResult(0);
          int offset = vInfo[out].size - vInfo[in].size;
          outInVals[out] = {in, offset};
          vInfo[in].size += offset;
          vInfo[in].lastUsed = vInfo[out].lastUsed;
        }
      }

      if (llvm::isa<Conv2DV2Op>(o)) {
        if (operationIds[o] == maxOpId) {
          auto convOp = dyn_cast<Conv2DV2Op>(o);
          auto in = o->getOperand(0);
          auto out = o->getResult(0);
          int offset = out.getType().dyn_cast<RankedTensorType>().getDimSize(
              3); // pixel size

          // since pad is input to this conv and already overlapped
          if (outInVals.count(in)) {
            // find the original input op
            auto firstVal = outInVals[in].first;
            auto firstOffset = outInVals[in].second;

            offset += vInfo[out].size - vInfo[firstVal].size;

            outInInVals[out] = {{in, offset}, {firstVal, firstOffset}};
            vInfo[firstVal].size += offset;
            vInfo[firstVal].lastUsed = vInfo[out].lastUsed;
          }
        }
      }
    }
  }

  // The comparator keeps the buffers ordered by id if their sizes are the same
  auto DecreasingSizesComparator = [&](QueueItem &lhs, QueueItem &rhs) {
    if (lhs.second != rhs.second) {
      return lhs.second < rhs.second;
    }
    return vInfo[lhs.first].id < vInfo[rhs.first].id;
  };
  // The top item is the largest one.
  llvm::PriorityQueue<QueueItem, std::vector<QueueItem>,
                      decltype(DecreasingSizesComparator)>
      queue(DecreasingSizesComparator);

  // Insert values and their sizes into priority queue
  for (auto v : values) {
    if (!outInVals.count(v) && !outInInVals.count(v) && !vInfo[v].isConstant) {
      queue.push({v, vInfo[v].size});
    }
  }

  ValuesOrderedByOffset allocatedValues;
  auto v = queue.top().first;
  queue.pop();
  allocatedValues.insert({v, 0});

  while (!queue.empty()) {
    auto v = queue.top().first;
    auto size = queue.top().second;
    queue.pop();

    // check with allocatedValues list
    int newOffset = getOffset(v, size, vInfo, allocatedValues);
    allocatedValues.insert({v, newOffset});
  }

  // Patch up overlapped buffers
  for (auto val : outInInVals) {
    auto out = val.first;
    auto inPair = val.second.first;
    auto firstValPair = val.second.second;

    auto in = inPair.first;
    auto offset = inPair.second;
    // We allocate here itself
    if (outInVals.count(in)) {
      outInVals.erase(in);
    }

    auto firstVal = firstValPair.first;
    auto firstOffset = firstValPair.second;

    auto it =
        std::find_if(allocatedValues.begin(), allocatedValues.end(),
                     [&](const QueueItem &p) { return p.first == firstVal; });

    if (it != allocatedValues.end()) {
      int currentOffset = it->second;
      allocatedValues.erase(it);
      allocatedValues.insert({firstVal, currentOffset + offset + firstOffset});
      allocatedValues.insert({in, currentOffset + offset});
      allocatedValues.insert({out, currentOffset});
    } else {
      assert(false);
    }
  }

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
    if (vInfo[v].isConstant) {
      allocatedValues.insert({v, -1});
    }
  }

  // Sort the allocated offsets by id, i.e., execution order
  auto cmp = [&](QueueItem a, QueueItem b) {
    return vInfo[a.first].id < vInfo[b.first].id;
  };
  std::multiset<QueueItem, decltype(cmp)> allocatedValuesOrderedByID(cmp);
  for (auto i : allocatedValues) {
    allocatedValuesOrderedByID.insert(i);
  }

  size_t peakUsed = 0;
  LLVM_DEBUG(llvm::dbgs() << "\nAllocated offsets : ");
  for (auto i : allocatedValuesOrderedByID) {
    offsets.push_back(i.second);
    peakUsed = std::max(peakUsed, vInfo[i.first].size + i.second);
    LLVM_DEBUG(llvm::dbgs() << "\nValue " << vInfo[i.first].id << ", size "
                            << vInfo[i.first].size << ", offset " << i.second);
  }
  LLVM_DEBUG(llvm::dbgs() << "\n\nPEAK USED : " << peakUsed << "\n\n");
  LLVM_DEBUG(llvm::dbgs() << "\n\n");
  peakMemoryUsed = peakUsed;

  return offsets;
}

} // namespace xcore
} // namespace mlir
