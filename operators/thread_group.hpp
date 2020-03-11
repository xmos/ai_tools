// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_THREAD_GROUP_H_
#define XCORE_THREAD_GROUP_H_

#ifndef _Bool
  #define _Bool int  // TODO: FIXME when new toolchain build is released
#endif

#ifdef XCORE 
    extern "C" {
        #include <xcore/thread.h>
    }
    namespace xcore {

    class ThreadGroup
    {
    public:
        ThreadGroup() {
            this->group_ = thread_group_alloc(); 
        }
        ~ThreadGroup() {
            thread_group_free(this->group_);
        }

        void Add(thread_function_t function, void* argument, size_t stack_words, void* stack) {
            thread_group_add(this->group_, function, argument, stack_base(stack, stack_words));
        }

        void Start() {
            thread_group_start(this->group_);
        }

        void Wait() {
            thread_group_wait(this->group_);
        }

    private:
        threadgroup_t group_;

    };

    } // namespace xcore
#else
    #include <thread>
    #include <vector>

    typedef void (*thread_function_t)(void *);
    
    class ThreadGroup
    {
    public:
        ThreadGroup() {}
        ~ThreadGroup() {}

        void Add(thread_function_t function, void* argument, size_t stack_words, void* stack) {
            group_.emplace_back(std::make_pair(function, argument));
        }

        void Start() {
            for(const auto& func_pair: group_) {
                threads_.push_back(std::thread(func_pair.first, func_pair.second));
            }
        }

        void Wait() {
            for(auto& thread: threads_) {
                thread.join();
            }
            threads_.clear();
        }

    private:
        std::vector<std::pair<thread_function_t, void*>> group_;
        std::vector<std::thread> threads_;
    };

#endif


#endif  // XCORE_THREAD_GROUP_H_
