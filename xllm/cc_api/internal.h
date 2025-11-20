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

#include "core/framework/request/request_output.h"
#include "core/framework/request/request_params.h"
#include "core/runtime/llm_master.h"
#include "types.h"

namespace xllm {

struct LLMCore {
  xllm::LLMMaster* master;
};

namespace utils {
xllm::RequestParams transfer_request_params(
    const XLLM_RequestParams& request_params) {
  xllm::RequestParams xllm_request_params;

  if (request_params.echo.has_value()) {
    xllm_request_params.echo = request_params.echo.value();
  }

  if (request_params.offline.has_value()) {
    xllm_request_params.offline = request_params.offline.value();
  }

  if (request_params.logprobs.has_value()) {
    xllm_request_params.logprobs = request_params.logprobs.value();
  }

  if (request_params.ignore_eos.has_value()) {
    xllm_request_params.ignore_eos = request_params.ignore_eos.value();
  }

  if (request_params.skip_special_tokens.has_value()) {
    xllm_request_params.skip_special_tokens =
        request_params.skip_special_tokens.value();
  }

  if (request_params.n.has_value()) {
    xllm_request_params.n = request_params.n.value();
  }

  if (request_params.max_tokens.has_value()) {
    xllm_request_params.max_tokens = request_params.max_tokens.value();
  }

  if (request_params.best_of.has_value()) {
    xllm_request_params.best_of = request_params.best_of.value();
  }

  if (request_params.slo_ms.has_value()) {
    xllm_request_params.slo_ms = request_params.slo_ms.value();
  }

  if (request_params.top_k.has_value()) {
    xllm_request_params.top_k = request_params.top_k.value();
  }

  if (request_params.top_p.has_value()) {
    xllm_request_params.top_p = request_params.top_p.value();
  }

  if (request_params.top_k.has_value()) {
    xllm_request_params.top_k = request_params.top_k.value();
  }

  if (request_params.frequency_penalty.has_value()) {
    xllm_request_params.frequency_penalty =
        request_params.frequency_penalty.value();
  }

  if (request_params.presence_penalty.has_value()) {
    xllm_request_params.presence_penalty =
        request_params.presence_penalty.value();
  }

  if (request_params.repetition_penalty.has_value()) {
    xllm_request_params.repetition_penalty =
        request_params.repetition_penalty.value();
  }

  if (request_params.service_request_id.has_value()) {
    xllm_request_params.service_request_id =
        request_params.service_request_id.value();
  }

  if (request_params.stop.has_value()) {
    xllm_request_params.stop = request_params.stop;
  }

  if (request_params.stop_token_ids.has_value()) {
    xllm_request_params.stop_token_ids = request_params.stop_token_ids;
  }

  if (request_params.streaming.has_value()) {
    const size_t best_of_value =
        xllm_request_params.best_of.value_or(xllm_request_params.n);
    if (request_params.streaming.value() &&
        best_of_value == xllm_request_params.n) {
      xllm_request_params.streaming = true;
    } else {
      xllm_request_params.streaming = false;
    }
  }

  if (request_params.beam_width.has_value()) {
    xllm_request_params.beam_width = request_params.beam_width.value();
    if (xllm_request_params.beam_width > 1) {
      xllm_request_params.ignore_eos = true;
    }
  }

  return xllm_request_params;
}

XLLM_Response build_xllm_response(xllm::RequestOutput output,
                                  const std::string& request_id,
                                  int64_t created_time,
                                  const std::string& model) {
  XLLM_Response response;

  response.id = request_id;
  response.object = "text_completion";
  response.created = created_time;
  response.model = model;

  response.choices.reserve(output.outputs.size());
  for (const auto& output : output.outputs) {
    XLLM_Choice choice;
    choice.index = output.index;
    choice.text = output.text;

    if (output.logprobs.has_value()) {
      for (const auto& logprob : output.logprobs.value()) {
        choice.logprobs.tokens.emplace_back(logprob.token);
        choice.logprobs.token_ids.emplace_back(logprob.token_id);
        choice.logprobs.token_logprobs.emplace_back(logprob.logprob);
      }
    }

    if (output.finish_reason.has_value()) {
      choice.finish_reason = output.finish_reason.value();
    }

    response.choices.emplace_back(choice);
  }

  return response;
}
}  // namespace utils
}  // namespace xllm