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

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "framework/model/model_input_params.h"
#include "framework/model_loader.h"
#include "models/model_registry.h"
#include "util/env_var.h"
#include "util/timer.h"

namespace xllm {

RecWorkerImpl::LlmRecWorkPipeline::LlmRecWorkPipeline(RecWorkerImpl& worker)
    : worker_(worker) {}

bool RecWorkerImpl::LlmRecWorkPipeline::create_model(RecWorkerImpl& worker,
                                                     ModelContext& context) {
  return worker.LLMWorkerImpl::init_model(context);
}

ForwardInput RecWorkerImpl::LlmRecWorkPipeline::prepare_inputs(Batch& batch) {
  return worker_.WorkerImpl::prepare_inputs(batch);
}

void RecWorkerImpl::LlmRecWorkPipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  worker_.WorkerImpl::prepare_work_before_execute(inputs, processed_inputs);
  // LlmRecDefault (pure qwen3) does not process mm_data.
  // For mm_data processing, use LlmRecWithMmDataWorkPipeline.
}

std::optional<ForwardOutput> RecWorkerImpl::LlmRecWorkPipeline::step(
    const ForwardInput& input) {
  return worker_.LLMWorkerImpl::step(input);
}

RecWorkerImpl::OneRecWorkPipeline::OneRecWorkPipeline(RecWorkerImpl& worker)
    : worker_(worker) {}

bool RecWorkerImpl::OneRecWorkPipeline::create_model(RecWorkerImpl& worker,
                                                     ModelContext& context) {
  // OneRec also uses LLM model for now, can be extended to create_rec_model
  // later
  return worker.LLMWorkerImpl::init_model(context);
}

ForwardInput RecWorkerImpl::OneRecWorkPipeline::prepare_inputs(Batch& batch) {
  ThreadPool* thread_pool = worker_.input_builder_thread_pool_
                                ? worker_.input_builder_thread_pool_.get()
                                : nullptr;

  return batch.prepare_rec_forward_input(worker_.options_.num_decoding_tokens(),
                                         /*min_decoding_batch_size=*/0,
                                         worker_.context_.get_model_args(),
                                         thread_pool);
}

void RecWorkerImpl::OneRecWorkPipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  worker_.WorkerImpl::prepare_work_before_execute(inputs, processed_inputs);
}

std::optional<ForwardOutput> RecWorkerImpl::OneRecWorkPipeline::step(
    const ForwardInput& input) {
  Timer timer;
  worker_.device_.set_device();

  const auto& sampling_params = input.sampling_params;
  const auto& input_params = input.input_params;

  const auto* onerec_params = input_params.onerec_params();
  CHECK(onerec_params != nullptr) << "OneRec requires rec_params.";

  const OneRecModelInputParams& rec_params = *onerec_params;

  torch::Tensor hidden_states;
  if (rec_params.rec_stage == OneRecModelInputParams::RecStage::PREFILL) {
    if (!rec_params.is_first_prefill) {
      ModelInputParams decoder_params = input_params;
      decoder_params.mutable_onerec_params().is_encoder_forward = false;
      hidden_states = worker_.model_executor_->forward(
          input.token_ids, input.positions, worker_.kv_caches_, decoder_params);
    } else {
      const bool has_sparse_embedding =
          rec_params.encoder_sparse_embedding.defined();
      const bool has_encoder_tokens = rec_params.encoder_token_ids.defined() &&
                                      rec_params.encoder_positions.defined();

      if (!has_sparse_embedding && !has_encoder_tokens) {
        LOG(ERROR) << "OneRec first prefill requires encoder inputs.";
        return std::nullopt;
      }

      ModelInputParams encoder_params = input_params;
      auto& mutable_onerec_params = encoder_params.mutable_onerec_params();
      mutable_onerec_params.is_encoder_forward = true;

      torch::Tensor encoder_tokens;
      if (has_sparse_embedding) {
        mutable_onerec_params.is_hybrid_mode = true;
        encoder_tokens = rec_params.encoder_sparse_embedding;
      } else {
        encoder_tokens = rec_params.encoder_token_ids;
      }

      worker_.model_executor_->forward(encoder_tokens,
                                       rec_params.encoder_positions,
                                       worker_.kv_caches_,
                                       encoder_params);

      ModelInputParams decoder_params = input_params;
      decoder_params.mutable_onerec_params().is_encoder_forward = false;
      hidden_states = worker_.model_executor_->forward(
          input.token_ids, input.positions, worker_.kv_caches_, decoder_params);
    }
  } else {
    ModelInputParams decoder_params = input_params;
    decoder_params.mutable_onerec_params().is_encoder_forward = false;
    hidden_states = worker_.model_executor_->forward(
        input.token_ids, input.positions, worker_.kv_caches_, decoder_params);
  }

  if (!hidden_states.defined()) {
    return std::nullopt;
  }

  if (!worker_.enable_schedule_overlap() && !worker_.driver_ &&
      !worker_.dp_driver_ && !worker_.options_.enable_speculative_decode()) {
    worker_.device_.synchronize_default_stream();
    COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
    DeviceMonitor::get_instance().update_active_activation_memory(
        worker_.device_.index());
    return std::nullopt;
  }

  torch::Tensor logits;
  if (sampling_params.selected_token_idxes.defined()) {
    logits = worker_.model_->logits(hidden_states,
                                    sampling_params.selected_token_idxes);
  }

  ForwardOutput output;

  if (sampling_params.selected_token_idxes.defined()) {
    auto sample_output = worker_.sampler_->forward(logits, sampling_params);
    output.logits = logits;
    output.sample_output = sample_output;
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
  }

  worker_.device_.synchronize_default_stream();
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      worker_.device_.index());

  return output;
}

// ============================================================
// LlmRecWithMmDataWorkPipeline Implementation (qwen3 with embedding)
// ============================================================

RecWorkerImpl::LlmRecWithMmDataWorkPipeline::LlmRecWithMmDataWorkPipeline(
    RecWorkerImpl& worker)
    : worker_(worker) {}

bool RecWorkerImpl::LlmRecWithMmDataWorkPipeline::create_model(
    RecWorkerImpl& worker,
    ModelContext& context) {
  return worker.LLMWorkerImpl::init_model(context);
}

ForwardInput RecWorkerImpl::LlmRecWithMmDataWorkPipeline::prepare_inputs(
    Batch& batch) {
  return worker_.WorkerImpl::prepare_inputs(batch);
}

void RecWorkerImpl::LlmRecWithMmDataWorkPipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  worker_.WorkerImpl::prepare_work_before_execute(inputs, processed_inputs);

  if (!inputs.input_params.mm_data.valid()) {
    return;
  }

  torch::Tensor multi_modal_values;
  torch::Tensor multi_modal_indices;

  const auto& processed_mm_data = processed_inputs.input_params.mm_data;
  if (auto res = processed_mm_data.get<torch::Tensor>("MULTI_MODAL_VALUES")) {
    multi_modal_values = res.value();
  }

  if (auto res = processed_mm_data.get<torch::Tensor>("MULTI_MODAL_INDICES")) {
    multi_modal_indices = res.value();
  }

  if (!multi_modal_values.defined() || !multi_modal_indices.defined()) {
    return;
  }

#if defined(USE_NPU)
  layer::NpuWordEmbedding npu_word_embedding = worker_.get_npu_word_embedding();
  torch::Tensor input_tokens_embedding =
      npu_word_embedding(processed_inputs.token_ids, 0);
#else
  layer::WordEmbedding word_embedding = worker_.get_word_embedding();
  torch::Tensor input_tokens_embedding =
      word_embedding->forward(processed_inputs.token_ids);
#endif

  std::vector<torch::indexing::TensorIndex> indices = {
      torch::indexing::TensorIndex(multi_modal_indices),
      torch::indexing::Slice()};
  input_tokens_embedding.index_put_(indices, multi_modal_values);

  processed_inputs.input_params.input_embedding = input_tokens_embedding;

  return;
}

std::optional<ForwardOutput> RecWorkerImpl::LlmRecWithMmDataWorkPipeline::step(
    const ForwardInput& input) {
  return worker_.LLMWorkerImpl::step(input);
}

RecWorkerImpl::RecWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : LLMWorkerImpl(parallel_args, device, options) {
  if (!is_driver()) {
    return;
  }

  const int64_t num_threads = std::max<int64_t>(
      1, util::get_int_env("XLLM_REC_INPUT_BUILDER_THREADS", 16));
  input_builder_thread_pool_ =
      std::make_shared<ThreadPool>(static_cast<size_t>(num_threads));
}

bool RecWorkerImpl::init_model(ModelContext& context) {
  const auto& model_type = context.get_model_args().model_type();
  rec_model_kind_ = get_rec_model_kind(model_type);
  CHECK(rec_model_kind_ != RecModelKind::kNone)
      << "Unsupported rec model_type: " << model_type;

  // Create work pipeline first
  auto pipeline_type = get_rec_pipeline_type(rec_model_kind_);
  work_pipeline_ = create_pipeline(pipeline_type, *this);

  // Let pipeline create model
  return work_pipeline_->create_model(*this, context);
}

ForwardInput RecWorkerImpl::prepare_inputs(Batch& batch) {
  CHECK(work_pipeline_ != nullptr) << "RecWorkerImpl is not initialized.";
  return work_pipeline_->prepare_inputs(batch);
}

void RecWorkerImpl::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  CHECK(work_pipeline_ != nullptr) << "RecWorkerImpl is not initialized.";
  work_pipeline_->prepare_work_before_execute(inputs, processed_inputs);
}

torch::Tensor RecWorkerImpl::merge_embeddings_by_indices(
    const torch::Tensor& input_tokens_embedding,
    const torch::Tensor& input_embedding,
    const std::vector<int64_t>& input_indices) {
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

  std::vector<int64_t> input_embedding_indices;
  for (int64_t i = 0; i < total_rows; ++i) {
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

std::optional<ForwardOutput> RecWorkerImpl::step(const ForwardInput& input) {
  CHECK(work_pipeline_ != nullptr) << "RecWorkerImpl is not initialized.";
  return work_pipeline_->step(input);
}

// ============================================================
// RecWorkerImpl pipeline factory (static method)
// ============================================================
std::unique_ptr<RecWorkerImpl::RecWorkPipeline> RecWorkerImpl::create_pipeline(
    RecPipelineType type,
    RecWorkerImpl& worker) {
  switch (type) {
    case RecPipelineType::kLlmRecDefault:
      return std::make_unique<LlmRecWorkPipeline>(worker);
    case RecPipelineType::kLlmRecWithMmData:
      return std::make_unique<LlmRecWithMmDataWorkPipeline>(worker);
    case RecPipelineType::kOneRecDefault:
      return std::make_unique<OneRecWorkPipeline>(worker);
    default:
      LOG(FATAL) << "Unknown RecWorkerImpl pipeline type: "
                 << static_cast<int>(type);
      return nullptr;
  }
}

}  // namespace xllm
