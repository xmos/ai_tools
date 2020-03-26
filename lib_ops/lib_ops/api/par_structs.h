// Copyright (c) 2020, XMOS Ltd, All rights reserved

#ifndef XCORE_PAR_STRUCTS_H_
#define XCORE_PAR_STRUCTS_H_

namespace xcore {

struct ParRegion {
  int32_t top;
  int32_t left;
  int32_t rows;
  int32_t cols;
  ParRegion(int32_t t, int32_t l, int32_t r, int32_t c)
      : top(t), left(l), rows(r), cols(c) {}
};

typedef std::vector<ParRegion> ParPlan;

}  // namespace xcore

#endif  // XCORE_PAR_STRUCTS_H_