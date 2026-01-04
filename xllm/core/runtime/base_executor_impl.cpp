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

#include "base_executor_impl.h"

#include <glog/logging.h>

#include "common/metrics.h"

namespace xllm {

BaseExecutorImpl::BaseExecutorImpl(CausalLM* model,
                                   const ModelArgs& args,
                                   const torch::Device& device,
                                   const runtime::Options& options)
    : model_(model), args_(args), device_(device), options_(options) {
  for (int i = 0; i < 10; i++) {
    auto stream = std::make_unique<Stream>(100);
    streams_.push_back(std::move(stream));
  }
}

ForwardInput BaseExecutorImpl::prepare_inputs(Batch& batch) {
  std::thread::id current_thread_id = std::this_thread::get_id();
  std::hash<std::thread::id> tid_hasher;
  size_t tid_hash = tid_hasher(current_thread_id);
  int index = static_cast<int>(tid_hash % static_cast<size_t>(10));

  c10::StreamGuard stream_guard = streams_[index]->set_stream_guard();

  auto input =
      batch.prepare_forward_input(options_.num_decoding_tokens(), 0, args_);

  streams_[index]->synchronize();

  return input;
}

torch::Tensor BaseExecutorImpl::run(const torch::Tensor& tokens,
                                    const torch::Tensor& positions,
                                    std::vector<KVCache>& kv_caches,
                                    const ModelInputParams& params) {
  COUNTER_INC(num_model_execution_total_eager);
  std::thread::id current_thread_id = std::this_thread::get_id();
  LOG(INFO) << "model forward at thread[" << current_thread_id << "]";
  std::hash<std::thread::id> tid_hasher;
  size_t tid_hash = tid_hasher(current_thread_id);
  int index = static_cast<int>(tid_hash % static_cast<size_t>(10));

  c10::StreamGuard stream_guard = streams_[index]->set_stream_guard();

  auto tensor = model_->forward(tokens, positions, kv_caches, params);

  streams_[index]->synchronize();

  return tensor;
}

}  // namespace xllm
