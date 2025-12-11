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

#include <unistd.h>

#include <chrono>
#include <iostream>
#include <optional>
#include <random>

#include "llm.h"

/**
 * How to compile examples.cpp:
 * Step 1: compile libxllm.so using the python setup.py build --generate-so true
 * command
 * Step 2: install the xllm package to the /usr/local directory using
 * install.sh
 * Step 3: g++ xllm/cc_api/examples.cpp -o llm_examples
 * -I/usr/local/xllm/include -L/usr/local/xllm/lib -lxllm
 * -Wl,-rpath=/usr/local/xllm/lib -D_GLIBCXX_USE_CXX11_ABI=0
 */

std::string devices = "npu:14";
std::string model_path = "/export/home/models/Qwen3-8B";

std::vector<std::vector<float>> generate_continuous_random_2d_vector(
    size_t rows,
    size_t cols,
    float min = 0.0f,
    float max = 1.0f) {
  unsigned int seed = static_cast<unsigned int>(
      std::chrono::system_clock::now().time_since_epoch().count());
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(min, max);

  std::vector<float> flat_data(rows * cols);
  for (auto& val : flat_data) {
    val = dist(gen);
  }

  std::vector<std::vector<float>> result;
  result.reserve(rows);
  for (size_t i = 0; i < rows; ++i) {
    result.emplace_back(flat_data.data() + i * cols,
                        flat_data.data() + (i + 1) * cols);
  }

  return result;
}

int main(int argc, char** argv) {
  xllm::LLM llm;
  xllm::XLLM_InitLLMOptions options;
  options.task = "rec";
  bool ret = llm.Initialize(model_path, devices, options);
  if (!ret) {
    std::cout << "llm init failed" << std::endl;
    return -1;
  }

  std::cout << "llm init succefully" << std::endl;

  if (false) {
    std::cout << "llm completions start" << std::endl;

    std::string prompt =
        "recommend 3 cheap and easy-to-use electric shavers, briefly "
        "describe the product name, price, and features";
    xllm::XLLM_RequestParams params;
    params.max_tokens = 500;
    xllm::XLLM_Response response =
        llm.Completions("Qwen3-8B", prompt, 20000, params);

    if (response.status_code != xllm::XLLM_StatusCode::kSuccess) {
      std::cout << "llm completions failed, error info:" << response.error_info
                << std::endl;
      return -1;
    } else {
      for (auto choice : response.choices) {
        if (choice.text.has_value()) {
          std::cout << "llm completions output:" << choice.text.value().c_str()
                    << std::endl;
        }
      }
    }

    std::cout << "llm completions end" << std::endl;
  }

  if (false) {
    std::cout << "llm chat completions start" << std::endl;
    xllm::XLLM_ChatMessage message;
    message.role = "user";
    message.content =
        "You are an expert in e-commerce scenarios. The current scenario is an "
        "e-commerce search engine with a comprehensive range of business "
        "categories. Your task is to determine whether 'user query' and "
        "'product title' are related in the e-commerce search engine. "
        "Discrimination criteria: If the search for 'user query' returns' "
        "product title 'that meets the user's needs, then the task is "
        "relevant. Output requirement: Please provide the answer in the "
        "'related' or 'unrelated' section, without mentioning any other "
        "content. User query: 'Hotpot sauce'. Product title: 'Grassland Red "
        "Sun Hotpot Base Dip Multi flavored Barbecue Sauce Tomato Sauce Leek "
        "Flower Sauce Nightsnack Paired with [New] Spicy Barbecue Sauce 100g'";

    std::vector<xllm::XLLM_ChatMessage> messages;
    messages.emplace_back(message);
    xllm::XLLM_RequestParams params;
    params.max_tokens = 100;

    xllm::XLLM_Response response =
        llm.ChatCompletions("Qwen3-8B", messages, 20000, params);

    if (response.status_code != xllm::XLLM_StatusCode::kSuccess) {
      std::cout << "llm completions failed, error info:" << response.error_info
                << std::endl;
      return -1;
    } else {
      for (auto choice : response.choices) {
        if (choice.message.has_value()) {
          std::cout << "llm completions output: role: "
                    << choice.message.value().role.c_str()
                    << ",content: " << choice.message.value().content.c_str()
                    << std::endl;
        }
      }
    }

    std::cout << "llm chat completions end" << std::endl;
  }

  if (true) {
    // std::optional<std::vector<std::vector<float>>> input_embedding =
    // generate_continuous_random_2d_vector(100, 4096);

    std::optional<std::vector<int>> input_tokens = std::vector<int>{
        66589, 220,   18,   11872, 323,  4135, 4686, 24673, 9072, 557, 21565,
        11,    26753, 7512, 279,   1985, 829,  11,   3349,  11,   323, 4419};

    std::optional<std::vector<int>> input_indices =
        std::vector<int>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};

    xllm::XLLM_RequestParams params;
    params.max_tokens = 100;

    xllm::XLLM_Response response = llm.BehaviorCompletions(
        "Qwen3-8B", input_tokens, input_indices, std::nullopt, 20000, params);

    if (response.status_code != xllm::XLLM_StatusCode::kSuccess) {
      std::cout << "llm completions failed, error info:" << response.error_info
                << std::endl;
      return -1;
    } else {
      for (auto choice : response.choices) {
        if (choice.text.has_value()) {
          std::cout << "llm completions output:" << choice.text.value().c_str()
                    << std::endl;
        }
        if (choice.logprobs.has_value()) {
          for (auto logprob : choice.logprobs.value()) {
            std::cout << "llm completions output2:" << logprob.token_id << ","
                      << logprob.token << std::endl;
          }
        } else {
          std::cout << "llm chat no logprobs end" << std::endl;
        }
      }
    }

    std::cout << "llm chat completions end" << std::endl;
  }

  sleep(10);

  return 0;
}