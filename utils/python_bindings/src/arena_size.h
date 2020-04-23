// Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
#ifndef ARENA_SIZE_H_
#define ARENA_SIZE_H_

#include <cstddef>

namespace xcore {

size_t search_tensor_arena_size(const char* model_content,
                                size_t model_content_size, size_t min_size,
                                size_t max_size);

}  // namespace xcore

#endif  // ARENA_SIZE_H_