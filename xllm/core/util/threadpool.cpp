/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "threadpool.h"

#include <thread>

#include "util/timer.h"

namespace xllm {
ThreadPool::ThreadPool(size_t num_threads) : ThreadPool(num_threads, nullptr) {}

ThreadPool::ThreadPool(size_t num_threads, Runnable init_func)
    : queues_(num_threads) {
  BlockingCounter counter(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    threads_.emplace_back([this,
                           i,
                           init_func_ptr = &init_func,
                           counter_ptr = &counter]() mutable {
      internal_loop(i, init_func_ptr, counter_ptr);
    });
  }
  running_sizes_.reserve(num_threads);
  runned_nums_.reserve(num_threads);
  runned_timers_.reserve(num_threads);

  counter.wait();
}

ThreadPool::~ThreadPool() {
  // push nullptr to the queue to signal threads to exit
  for (size_t i = 0; i < threads_.size(); ++i) {
    queues_[i].push(nullptr);
  }
  // wait for all threads to finish
  for (auto& thread : threads_) {
    thread.join();
  }
}

// schedule a runnable to be executed
int32_t ThreadPool::schedule(Runnable runnable) {
  if (runnable == nullptr) {
    return -1;
  }

  size_t current;
  size_t next;
  do {
    current = index_.load(std::memory_order_relaxed);
    next = (current + 1) % queues_.size();
  } while (!index_.compare_exchange_weak(
      current, next, std::memory_order_relaxed, std::memory_order_relaxed));
  queues_[current].push(std::move(runnable));
  return current;
}

void ThreadPool::schedule_with_tid(Runnable runnable, size_t tid) {
  if (runnable == nullptr) {
    return;
  }

  queues_[tid].push(std::move(runnable));
}

void ThreadPool::internal_loop(size_t index,
                               Runnable* init_func,
                               BlockingCounter* block_counter) {
  if (init_func != nullptr && *init_func != nullptr) {
    (*init_func)();
  }
  block_counter->decrement_count();

  while (true) {
    Runnable runnable = queues_[index].pop();

    if (runnable == nullptr) {
      // nullptr is a signal to exit
      break;
    }
    Timer timer;
    timer.reset();
    running_sizes_[index]++;
    runnable();
    running_sizes_[index]--;
    runned_nums_[index]++;
    runned_timers_[index] += timer.elapsed_milliseconds();
  }
}

int ThreadPool::waiting_runnable_size() {
  int count = 0;
  for (auto& queue : queues_) {
    count += queue.size();
  }

  return count;
}

int ThreadPool::running_runnable_size() {
  int count = 0;
  for (auto& running_size : running_sizes_) {
    count += running_size;
  }

  return count;
}

double ThreadPool::run_runnable_elapsed_time() {
  std::vector<double> times;
  times.reserve(runned_timers_.size());
  for (int i = 0; i < runned_timers_.size(); i++) {
    if (runned_nums_[i] == 0) return 0;
    times[i] = runned_timers_[i] / runned_nums_[i];
  }

  double total = 0;
  for (int i = 0; i < times.size(); i++) {
    total += times[i];
  }

  return total / times.size();
}

}  // namespace xllm
