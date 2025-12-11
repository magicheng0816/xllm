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

#include "rec_master.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "util/scope_guard.h"
#include "util/timer.h"

namespace xllm {
LLMRecMaster::LLMRecMaster(const Options& options) : LLMMaster(options) {}
LLMRecMaster::~LLMRecMaster() {}

void LLMRecMaster::handle_request(
    std::optional<std::vector<int>> input_tokens,
    std::optional<std::vector<int>> input_indices,
    std::optional<std::vector<std::vector<float>>> input_embedding,
    RequestParams sp,
    std::optional<Call*> call,
    OutputCallback callback) {
  scheduler_->incr_pending_requests(1);

  auto cb = [callback = std::move(callback),
             scheduler = scheduler_.get()](const RequestOutput& output) {
    output.log_request_status();
    return callback(output);
  };

  // add into the queue
  threadpool_->schedule([this,
                         input_tokens = std::move(input_tokens),
                         input_indices = std::move(input_indices),
                         input_embedding = std::move(input_embedding),
                         sp = std::move(sp),
                         callback = std::move(cb),
                         call]() mutable {
    AUTO_COUNTER(request_handling_latency_seconds_completion);

    // remove the pending request after scheduling
    SCOPE_GUARD([this] { scheduler_->decr_pending_requests(); });

    Timer timer;
    // verify the prompt
    if (!sp.verify_params(callback)) {
      return;
    }

    auto request = generate_request(std::move(input_tokens),
                                    std::move(input_indices),
                                    std::move(input_embedding),
                                    sp,
                                    call,
                                    callback);
    if (!request) {
      return;
    }

    if (!scheduler_->add_request(request)) {
      CALLBACK_WITH_ERROR(StatusCode::RESOURCE_EXHAUSTED,
                          "No available resources to schedule request");
    }
  });
}

std::shared_ptr<Request> LLMRecMaster::generate_request(
    std::optional<std::vector<int>> input_tokens,
    std::optional<std::vector<int>> input_indices,
    std::optional<std::vector<std::vector<float>>> input_embedding,
    const RequestParams& sp,
    std::optional<Call*> call,
    OutputCallback callback) {
  // encode the prompt
  Timer timer;

  std::vector<int> local_prompt_tokens;
  std::vector<int> local_input_tokens;
  std::vector<int> local_input_indices;
  torch::Tensor local_input_embedding;
  torch::Tensor local_input_tokens_tensor;
  torch::Tensor local_input_indices_tensor;

  if (input_tokens.has_value()) {
    local_input_tokens = std::move(input_tokens.value());
    local_input_indices = std::move(input_indices.value());

    if (local_input_tokens.size() > 0 && local_input_indices.size() > 0) {
      // Ensure independence from the input data after passing through the
      // master
      local_input_tokens_tensor =
          torch::from_blob(local_input_tokens.data(),
                           {local_input_tokens.size()},
                           torch::dtype(torch::kInt32).device(torch::kCPU))
              .clone();
      local_input_indices_tensor =
          torch::from_blob(local_input_indices.data(),
                           {local_input_indices.size()},
                           torch::dtype(torch::kInt32).device(torch::kCPU))
              .clone();
    }

    local_prompt_tokens = local_input_tokens;
  }

  if (input_embedding.has_value()) {
    std::vector<std::vector<float>> input_embedding_vec =
        input_embedding.value();

    if (0 == input_embedding_vec.size()) {
      LOG(ERROR) << "Input embedding is empty";
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Input embedding is empty");
      return nullptr;
    }

    if (input_embedding_vec[0].size() != model_args_.hidden_size()) {
      LOG(ERROR) << "Input embedding is invalid";
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Input embedding is invalid");
      return nullptr;
    }

    const int64_t rows = static_cast<int64_t>(input_embedding_vec.size());
    const int64_t cols = static_cast<int64_t>(input_embedding_vec[0].size());

    std::vector<float> flat_data;
    flat_data.reserve(rows * cols);
    for (const auto& row : input_embedding_vec) {
      flat_data.insert(flat_data.end(), row.begin(), row.end());
    }

    local_input_embedding =
        torch::from_blob(flat_data.data(),
                         {rows, cols},
                         torch::dtype(torch::kFloat32).device(torch::kCPU))
            .clone();

    const int kDefaultPlaceholderToken = 20152019;
    local_prompt_tokens.insert(local_prompt_tokens.end(),
                               local_input_embedding.size(0),
                               kDefaultPlaceholderToken);
  }

  if (local_prompt_tokens.empty()) {
    LOG(ERROR) << "Prompt is empty";
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "Prompt is empty");
    return nullptr;
  }

  if (local_input_indices.size() > 0) {
    if (local_input_tokens.size() != local_input_indices.size()) {
      LOG(ERROR) << "Input token positions is not match input tokens";
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Input token positions is not match input tokens");
      return nullptr;
    }

    int total_size = local_input_embedding.defined()
                         ? local_input_tokens.size() +
                               static_cast<int>(local_input_embedding.size(0))
                         : local_input_tokens.size();

    std::unordered_set<int> seen;
    for (size_t i = 0; i < local_input_indices.size(); ++i) {
      int index = local_input_indices[i];
      if (index < 0 || index >= total_size) {
        LOG(ERROR) << "Input token indices contain invalid values";
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "Input token indices contain invalid values");
        return nullptr;
      }

      if (seen.find(index) != seen.end()) {
        LOG(ERROR) << "Input token indices contain duplicate values";
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "Input token indices contain duplicate values");
        return nullptr;
      }
      seen.insert(index);
    }
  }

  MMData mm_data;
  if (local_input_tokens_tensor.defined()) {
    mm_data.add(
        MMType::EMBEDDING, LLM_REC_INPUT_TOKENS, local_input_tokens_tensor);
    mm_data.add(
        MMType::EMBEDDING, LLM_REC_INPUT_INDICES, local_input_indices_tensor);
  }
  if (local_input_embedding.defined()) {
    mm_data.add(
        MMType::EMBEDDING, LLM_REC_INPUT_EMBEDDING, local_input_embedding);
  }

  int32_t max_context_len = model_args_.max_position_embeddings();
  if (!options_.enable_chunked_prefill()) {
    max_context_len =
        std::min(max_context_len, options_.max_tokens_per_batch());
  }
  if (local_prompt_tokens.size() >= max_context_len) {
    LOG(ERROR) << "Prompt is too long: " << local_prompt_tokens.size();
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "Prompt is too long");
    return nullptr;
  }

  uint32_t max_tokens = sp.max_tokens;
  if (max_tokens == 0) {
    const uint32_t kDefaultMaxTokens = 5120;
    max_tokens = kDefaultMaxTokens;
  }

  // allocate enough capacity for prompt tokens, max tokens, and speculative
  // tokens
  size_t capacity = local_prompt_tokens.size() + max_tokens +
                    options_.num_speculative_tokens() + /*bouns_token*/ 1;
  if (options_.enable_schedule_overlap()) {
    capacity += options_.num_speculative_tokens() + 1;
  }
  const size_t best_of = sp.best_of.value_or(sp.n);

  RequestSamplingParam sampling_param;
  sampling_param.frequency_penalty = sp.frequency_penalty;
  sampling_param.presence_penalty = sp.presence_penalty;
  sampling_param.repetition_penalty = sp.repetition_penalty;
  sampling_param.temperature = sp.temperature;
  sampling_param.top_p = sp.top_p;
  sampling_param.top_k = sp.top_k;
  sampling_param.logprobs = sp.logprobs;
  sampling_param.top_logprobs = sp.top_logprobs;
  sampling_param.is_embeddings = sp.is_embeddings;
  sampling_param.beam_width = sp.beam_width;
  if (best_of > sp.n) {
    // enable logprobs for best_of to generate sequence logprob
    sampling_param.logprobs = true;
  }
  // sampling_param.do_sample = sp.do_sample;

  std::unordered_set<int32_t> stop_tokens;
  if (sp.stop_token_ids.has_value()) {
    const auto& stop_token_ids = sp.stop_token_ids.value();
    stop_tokens.insert(stop_token_ids.begin(), stop_token_ids.end());
  } else {
    stop_tokens = model_args_.stop_token_ids();
  }
  std::vector<std::vector<int32_t>> stop_sequences;
  if (sp.stop.has_value()) {
    for (const auto& s : sp.stop.value()) {
      std::vector<int> tmp_tokens;
      if (!tokenizer_->encode(s, &tmp_tokens)) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "Failed to encode stop sequence");
        LOG(ERROR) << "Failed to encode stop sequence: " << s;
        return nullptr;
      }
      stop_sequences.push_back(std::move(tmp_tokens));
    }
  }

  StoppingChecker stopping_checker(
      max_tokens,
      max_context_len - options_.num_speculative_tokens(),
      model_args_.eos_token_id(),
      sp.ignore_eos,
      std::move(stop_tokens),
      std::move(stop_sequences));

  bool stream = sp.streaming;
  // results cannot be streamed when best_of != n
  if (best_of != sp.n) {
    stream = false;
  }

  std::string empty_prompt = "";
  RequestState req_state(empty_prompt,
                         std::move(local_prompt_tokens),
                         std::move(mm_data),
                         std::move(sampling_param),
                         std::move(stopping_checker),
                         capacity,
                         sp.n,
                         best_of,
                         sp.logprobs,
                         stream,
                         sp.echo,
                         sp.skip_special_tokens,
                         options_.enable_schedule_overlap(),
                         callback,
                         nullptr,
                         sp.decode_address,
                         call);

  auto request = std::make_shared<Request>(sp.request_id,
                                           sp.x_request_id,
                                           sp.x_request_time,
                                           std::move(req_state),
                                           sp.service_request_id,
                                           sp.offline,
                                           sp.slo_ms,
                                           sp.priority);

  // add one sequence, rest will be added by scheduler
  return request;
}
}  // namespace xllm
