// Copyright 2023 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Analysis/MemoryPlan.h"
#include "IR/XCoreOps.h"
#include "Transforms/Options.h"
#include "Utils/Util.h"

#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "xcore-memory-plan"

namespace mlir::xcore {

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
    operationIds.insert({op, operationIds.size()});
    operations.push_back(op);

    if (op == funcOp || llvm::isa<quantfork::StatisticsOp>(op)) {
      return;
    }

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
  llvm::DenseMap<Value, std::pair<Value, int>> inOutMap;
  llvm::DenseSet<Operation *> alreadyVisited;
  if (overlapOps) {
    for (auto o : operations) {
      if (o->hasTrait<OpTrait::xcore::MemoryOverlappable>() &&
          !alreadyVisited.contains(o) && o->getOperand(0).hasOneUse()) {
        alreadyVisited.insert(o);

        llvm::SmallVector<Value> inputVals;
        auto inVal = o->getOperand(0);
        inputVals.push_back(inVal);

        auto outVal = o->getResult(0);
        auto nextOp = *outVal.getUsers().begin();
        // Identify chain of overlappable Ops
        while (outVal.hasOneUse() &&
               nextOp->hasTrait<OpTrait::xcore::MemoryOverlappable>()) {
          inVal = nextOp->getOperand(0);
          inputVals.push_back(inVal);
          alreadyVisited.insert(nextOp);
          outVal = nextOp->getResult(0);
          nextOp = *outVal.getUsers().begin();
        }

        // Set first Used of output Val to the first input Val
        vInfo[outVal].firstUsed = vInfo[inputVals[0]].firstUsed;
        auto unalignedSizeOutVal =
            utils::getShapedTypeSize(outVal.getType().dyn_cast<ShapedType>());
        size_t maxSizeNeeded = 0;
        for (auto inV : inputVals) {
          auto unalignedSizeInV =
              utils::getShapedTypeSize(inV.getType().dyn_cast<ShapedType>());
          auto unalignedOffset = unalignedSizeOutVal - unalignedSizeInV;
          // Align offset up to double word = 8 bytes
          auto offset = ((unalignedOffset + 7) / 8) * 8;
          maxSizeNeeded = std::max(vInfo[inV].size + offset, maxSizeNeeded);
          inOutMap[inV] = {outVal, offset};
        }
        // The aligned input val size plus aligned offset might be larger than
        // aligned output val size
        vInfo[outVal].size = std::max(vInfo[outVal].size, maxSizeNeeded);
      }
    }
  }

  // The comparator keeps the buffers ordered by id if their sizes are the
  // same
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
    if (!inOutMap.count(v) && !vInfo[v].isConstant) {
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
  for (auto val : inOutMap) {
    auto in = val.first;
    auto out = val.second.first;
    auto offset = val.second.second;

    auto it = std::find_if(allocatedValues.begin(), allocatedValues.end(),
                           [&](const QueueItem &p) { return p.first == out; });

    if (it != allocatedValues.end()) {
      int currentOffset = it->second;
      allocatedValues.insert({in, currentOffset + offset});
    } else {
      assert(false);
    }
  }

  // Insert -1 offset for constant values
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
    LLVM_DEBUG(llvm::dbgs() << "\nValue " << vInfo[i.first].id << ", size = "
                            << vInfo[i.first].size << ", offset = " << i.second
                            << ", first = " << vInfo[i.first].firstUsed
                            << ", last = " << vInfo[i.first].lastUsed);
  }
  LLVM_DEBUG(llvm::dbgs() << "\n\nPEAK USED : " << peakUsed << "\n\n");
  LLVM_DEBUG(llvm::dbgs() << "\n\n");
  peakMemoryUsed = peakUsed;

  return offsets;
}

} // namespace mlir::xcore
