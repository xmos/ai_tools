// Copyright (c) 2020, XMOS Ltd, All rights reserved

#ifndef XCORE_NN_OPERATORS_HPP_
#define XCORE_NN_OPERATORS_HPP_

#include <vector>

extern "C" {
    #include "nn_operator.h"
    #include "nn_types.h"
}
#include "thread_group.hpp"

#ifdef XCORE
    #define ATTRIBUTE_KERNEL_FUNCTION __attribute__((fptrgroup("kernel_function")))
#else
    #define ATTRIBUTE_KERNEL_FUNCTION
#endif

namespace xcore {

extern ThreadGroup *thread_group;

struct ParRegion{
    uint32_t top;
    uint32_t left;
    uint32_t rows;
    uint32_t cols;
    ParRegion(uint32_t t, uint32_t l, uint32_t r, uint32_t c)
      : top(t), left(l), rows(r), cols(c) {}
};

typedef std::vector<ParRegion> ParPlan;

} // namespace xcore


#endif  // XCORE_NN_OPERATORS_HPP_