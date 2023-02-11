// Copyright 2023 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#ifndef XFORMER_ANALYSIS_MEMORYPLANNER_H
#define XFORMER_ANALYSIS_MEMORYPLANNER_H

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/PriorityQueue.h"

#include <set>

namespace mlir {
namespace xcore {

/*

- Include liveness analysis
- Save array of op pointers along with index for each
- Create allocation info structure with liveness info of first used and last
used op
*/

class MemoryPlanner {
public:
  using QueueItem = std::pair<Value, size_t>;
  struct AscendingOffsetsComparator {
    bool operator()(const QueueItem &lhs, const QueueItem &rhs) const {
      return (lhs.second < rhs.second);
    }
  };

  using OrderedOffsets = std::multiset<QueueItem, AscendingOffsetsComparator>;

  struct ValueInfo {
    size_t id;
    size_t size;
    bool constant;
    int firstUsed;
    int lastUsed;
  };

  MemoryPlanner(func::FuncOp op);

  std::vector<int> getOffsets();

private:
  int getNewOffset(Value v, int size, OrderedOffsets &selected);

  DenseMap<Value, ValueInfo> valueInfo;

  std::vector<Value> values;

  // Maps each Operation to a unique ID according to the program sequence.
  DenseMap<Operation *, size_t> operationIds;

  // Stores all operations according to the program sequence.
  std::vector<Operation *> operations;

  Liveness liveness;

  func::FuncOp funcOp;
};

/*
- Memory planner structure with pointer to ops
- Put op and op sizes in max priority heap
- Greedy algorithm
- Alignment for buffers?

- How does export to flatbuffer work?
- How to import offsets in tflite-micro?
- Do we need to handle constant buffers also?


- Reorder graph optimization
Associative Instruction Reordering to Alleviate Register Pressure
https://hal.inria.fr/hal-01956260/document

*/

} // namespace xcore
} // namespace mlir

#endif // XFORMER_ANALYSIS_MEMORYPLANNER_H
