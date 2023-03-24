// Copyright 2023 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#ifndef XFORMER_ANALYSIS_MEMORYPLAN_H
#define XFORMER_ANALYSIS_MEMORYPLAN_H

#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/PriorityQueue.h"

#include <set>

namespace mlir {
namespace xcore {

// Represents an analysis for memory planning of a given FuncOp for a model.
// - Uses liveness analysis and a greedy algorithm to arrange buffers in memory.
// - Tries to overlap input and output buffers based on the op characteristics.
// - Calculates maximum width of the network for planning operator splitting.

// Future optimizations to consider
// - Reorder graph optimization
// Associative Instruction Reordering to Alleviate Register Pressure
// https://hal.inria.fr/hal-01956260/document

class MemoryPlan {
public:
  MemoryPlan(Operation *op);

  // The offset allocation algorithm is similar in implementation to the greedy
  // memory planner in tflite-micro. The algorithm works like this:
  //  - The buffers are sorted in descending order of size. A PriorityQueue is
  //  used for this.
  //  - The largest buffer is allocated at offset zero.
  //  - The rest of the buffers are popped from the queue in descending size
  //  order.
  //  - Every popped buffer is compared with the already allocated buffers.
  //  - The first gap between simultaneously active buffers that the current
  //    buffer fits into will be used.
  //  - If no large-enough gap is found, the current buffer is placed after the
  //    last buffer that's simultaneously active.
  //  - This continues until all buffers are placed, and the offsets stored.
  std::vector<int> getAllocatedOffsets();

  int getMaxMemoryUsed();

  // OpSplitPlan getOpSplitPlan();

private:
  /// Initializes the internal mappings.
  void build();

  using QueueItem = std::pair<Value, size_t>;
  //
  struct IncreasingOffsetsComparator {
    bool operator()(const QueueItem &lhs, const QueueItem &rhs) const {
      return (lhs.second < rhs.second);
    }
  };
  //
  using ValuesOrderedByOffset =
      std::multiset<QueueItem, IncreasingOffsetsComparator>;

  struct ValueInfo {
    size_t id;
    size_t size;
    bool isConstant;
    int firstUsed;
    int lastUsed;
  };

  int getOffset(Value v, int size, ValuesOrderedByOffset &allocatedOffsets);

  DenseMap<Value, ValueInfo> valueInfo;

  std::vector<Value> values;

  // Maps each Operation to a unique ID according to the program sequence.
  DenseMap<Operation *, size_t> operationIds;

  // Stores all operations according to the program sequence.
  std::vector<Operation *> operations;

  Liveness liveness;

  Operation *op;
};

} // namespace xcore
} // namespace mlir

#endif // XFORMER_ANALYSIS_MEMORYPLAN_H
