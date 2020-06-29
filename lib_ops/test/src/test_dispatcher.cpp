// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/dispatcher.h"
#include "unity.h"
#include "unity_fixture.h"

struct ThreadData {
  int value;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void thread_worker(void *context) {
  ThreadData *data = (ThreadData *)context;
  data->value++;
}
}

TEST_GROUP(dispatcher);

TEST_SETUP(dispatcher) {}

TEST_TEAR_DOWN(dispatcher) {}

TEST(dispatcher, test_current_core) {
  size_t buffer_size = 1000;
  void *buffer[buffer_size];
  int num_cores = 5;
  size_t stack_words = 0;

  xcore::Dispatcher dispatcher(buffer, buffer_size, true);
  GET_STACKWORDS(stack_words, thread_worker);
  dispatcher.InitializeTasks(thread_worker, stack_words + 2);

  ThreadData data[num_cores];
  for (int i = 0; i < num_cores; i++) {
    data[i].value = i;
    dispatcher.AddTask(reinterpret_cast<void *>(&data[i]));
  }
  dispatcher.JoinTasks();

  for (int i = 0; i < num_cores; i++) {
    TEST_ASSERT_EQUAL_INT(i + 1, data[i].value);
  }
}

TEST(dispatcher, test_not_current_core) {
  size_t buffer_size = 1000;
  void *buffer[buffer_size];
  int num_cores = 4;
  size_t stack_words = 0;

  xcore::Dispatcher dispatcher(buffer, buffer_size, true);
  GET_STACKWORDS(stack_words, thread_worker);
  dispatcher.InitializeTasks(thread_worker, stack_words + 2);

  ThreadData data[num_cores];
  for (int i = 0; i < num_cores; i++) {
    data[i].value = i;
    dispatcher.AddTask(reinterpret_cast<void *>(&data[i]));
  }
  dispatcher.JoinTasks();

  for (int i = 0; i < num_cores; i++) {
    TEST_ASSERT_EQUAL_INT(i + 1, data[i].value);
  }
}

TEST(dispatcher, test_reset) {
  size_t buffer_size = 1000;
  void *buffer[buffer_size];
  int num_cores = 3;
  size_t stack_words = 0;
  ThreadData data[num_cores];

  xcore::Dispatcher dispatcher(buffer, buffer_size, true);
  GET_STACKWORDS(stack_words, thread_worker);
  dispatcher.InitializeTasks(thread_worker, stack_words + 2);

  for (int iter = 0; iter < 100; iter++) {
    for (int i = 0; i < num_cores; i++) {
      data[i].value = i;
      dispatcher.AddTask(reinterpret_cast<void *>(&data[i]));
    }
    dispatcher.JoinTasks();

    for (int i = 0; i < num_cores; i++) {
      TEST_ASSERT_EQUAL_INT(i + 1, data[i].value);
    }

    dispatcher.Reset();
  }
}

TEST_GROUP_RUNNER(dispatcher) {
  RUN_TEST_CASE(dispatcher, test_not_current_core);
  RUN_TEST_CASE(dispatcher, test_current_core);
  RUN_TEST_CASE(dispatcher, test_reset);
}
