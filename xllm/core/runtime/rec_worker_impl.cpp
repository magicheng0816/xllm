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

#include "rec_worker_impl.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "core/layers/word_embedding.h"

namespace xllm {

LLMRecWorkerImpl::LLMRecWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options)
    : LLMWorkerImpl(parallel_args, device, options) {}

void LLMRecWorkerImpl::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  WorkerImpl::prepare_work_before_execute(inputs, processed_inputs);

  if (!inputs.input_params.mm_data.valid()) {
    return;
  }

  torch::Tensor input_embedding;
  torch::Tensor input_tokens_tensor;
  torch::Tensor input_indices_tensor;

  const auto& mm_data = inputs.input_params.mm_data;
  const auto& processed_mm_data = processed_inputs.input_params.mm_data;
  if (auto res = processed_mm_data.get<torch::Tensor>(
          std::string(LLM_REC_INPUT_TOKENS)))
    input_tokens_tensor = res.value();

  // Get the token indices tensor on the host, as input_embedding indices need
  // to be generated based on it
  if (auto res = mm_data.get<torch::Tensor>(LLM_REC_INPUT_INDICES))
    input_indices_tensor = res.value();

  if (auto res = processed_mm_data.get<torch::Tensor>(LLM_REC_INPUT_EMBEDDING))
    input_embedding = res.value();

  if (input_embedding.defined()) input_embedding = input_embedding.to(dtype_);

  // The verification of
  // input_indices_tensor/input_tokens_tenser/input_imbedding has already been
  // completed in LLMRecMaster, so no other verification will be performed here
  // except for the definite() verification
  if (input_indices_tensor.defined()) {
    layer::WordEmbedding word_embedding = get_word_embedding();
    torch::Tensor input_tokens_embedding =
        word_embedding(input_tokens_tensor, 0);

    if (input_embedding.defined()) {
      std::vector<int> input_indices(
          input_indices_tensor.data_ptr<int>(),
          input_indices_tensor.data_ptr<int>() + input_indices_tensor.numel());

      processed_inputs.input_params.input_embedding =
          merge_embeddings_by_indices(
              input_tokens_embedding, input_embedding, input_indices);
    } else {
      processed_inputs.input_params.input_embedding = input_tokens_embedding;
    }
  } else if (input_embedding.defined()) {
    processed_inputs.input_params.input_embedding = input_embedding;
  }
}

torch::Tensor LLMRecWorkerImpl::merge_embeddings_by_indices(
    const torch::Tensor& input_tokens_embedding,
    const torch::Tensor& input_embedding,
    const std::vector<int>& input_indices) {
  CHECK_EQ(input_embedding.dim(), 2);
  CHECK_EQ(input_tokens_embedding.dim(), 2);
  CHECK_EQ(input_tokens_embedding.size(1), input_embedding.size(1));
  CHECK_EQ(input_tokens_embedding.dtype(), input_embedding.dtype());
  CHECK_EQ(input_tokens_embedding.device(), input_embedding.device());

  const int64_t total_rows =
      input_tokens_embedding.size(0) + input_embedding.size(0);
  const int64_t cols = input_embedding.size(1);

  torch::Device device = input_embedding.device();
  torch::Tensor merged = torch::empty(
      {total_rows, cols}, torch::dtype(input_embedding.dtype()).device(device));

  std::vector<int> input_embedding_indices;
  for (int i = 0; i < static_cast<int>(total_rows); ++i) {
    if (std::find(input_indices.begin(), input_indices.end(), i) ==
        input_indices.end()) {
      input_embedding_indices.push_back(i);
    }
  }

  CHECK_EQ(input_embedding_indices.size(), input_embedding.size(0));

  torch::Tensor input_embedding_indices_tensor =
      torch::tensor(input_embedding_indices, torch::kInt64).to(device);
  merged.index_put_({input_embedding_indices_tensor, torch::indexing::Ellipsis},
                    input_embedding);

  torch::Tensor input_indices_tensor =
      torch::tensor(input_indices, torch::kInt64).to(device);
  merged.index_put_({input_indices_tensor, torch::indexing::Ellipsis},
                    input_tokens_embedding);

  return merged;
}

}  // namespace xllm