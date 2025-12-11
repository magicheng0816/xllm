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

#pragma once

#include <torch/torch.h>

#include "llm_worker_impl.h"

namespace xllm {
class LLMRecWorkerImpl : public LLMWorkerImpl {
 public:
  LLMRecWorkerImpl(const ParallelArgs& parallel_args,
                   const torch::Device& device,
                   const runtime::Options& options);

  virtual ~LLMRecWorkerImpl() = default;

  void prepare_work_before_execute(const ForwardInput& inputs,
                                   ForwardInput& processed_inputs) override;

 private:
  torch::Tensor merge_embeddings_by_indices(
      const torch::Tensor& input_tokens_embedding,
      const torch::Tensor& input_embedding,
      const std::vector<int>& input_indices);
};

}  // namespace xllm