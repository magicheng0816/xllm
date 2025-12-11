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

#include <vector>

#include "common/options.h"
#include "runtime/llm_master.h"

namespace xllm {

// A generative recommendation master based on LLM model
class LLMRecMaster : public LLMMaster {
 public:
  explicit LLMRecMaster(const Options& options);
  virtual ~LLMRecMaster();

  // behavior completions
  void handle_request(
      std::optional<std::vector<int>> input_tokens,
      std::optional<std::vector<int>> input_indices,
      std::optional<std::vector<std::vector<float>>> input_embedding,
      RequestParams sp,
      std::optional<Call*> call,
      OutputCallback callback);

 private:
  std::shared_ptr<Request> generate_request(
      std::optional<std::vector<int>> input_tokens,
      std::optional<std::vector<int>> input_indices,
      std::optional<std::vector<std::vector<float>>> input_embedding,
      const RequestParams& sp,
      std::optional<Call*> call,
      OutputCallback callback);
};

}  // namespace xllm