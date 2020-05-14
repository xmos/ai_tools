// Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
#ifndef ARENA_SIZE_H_
#define ARENA_SIZE_H_

#include <cstddef>

namespace xcore {

void search_arena_sizes(const char *model_content, size_t model_content_size,
                        size_t min_arena_size, size_t max_arena_size,
                        size_t *arena_size, size_t *heap_size);

}  // namespace xcore

#endif  // ARENA_SIZE_H_