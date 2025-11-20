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

#include <optional>
#include <string>

#include "macros.h"
#include "types.h"

namespace xllm {

// Forward declaration
typedef struct LLMCore LLMCore;
struct XLLM_RequestParams;
struct XLLM_InitLLMOptions;
struct XLLM_Response;

// A wrapper for loading, initializing, and text generation functions of large
// language models
class XLLM_CAPI_EXPORT LLM {
 public:
  LLM();
  virtual ~LLM();

  /**
   * @brief Initialize the model: Load model files and configure runtime
   * environment
   * @param model_path Path to model files
   * @param devices Device configuration (format: "npu:1" for specific NPU,
   * "auto" for auto-selection)
   * @param init_options Advanced initialization options, Provided default
   * configuration
   * @return bool true if initialization succeeds; false if fails
   * @note Must be called before Generate(), and only needs to be called once
   */
  bool Initialize(const std::string& model_path,
                  const std::string& devices,
                  const XLLM_InitLLMOptions& init_options);

  /**
   * @brief Text generation interface: Generate response based on input prompt
   * @param model_id ID of the model to use (must be loaded via Initialize() and
   * exist in model_ids_)
   * @param prompt Input text prompt
   * @param request_params Generation parameters, Provided default param values
   * @param callback Callback function for receiving results
   */
  void Generate(const std::string& model_id,
                const std::string& prompt,
                const XLLM_RequestParams& request_params,
                XLLM_OutputCallback callback);

 private:
  // Initialization state flag
  bool initialized_ = false;

  // Opaque pointer to core model logic
  LLMCore* llm_core_ = nullptr;

  // List of loaded model IDs
  std::vector<std::string> model_ids_;
};
}  // namespace xllm
