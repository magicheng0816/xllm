#include "llm.h"

#include <unistd.h>

#include <iostream>

int main(int argc, char** argv) {
  xllm::LLM llm;
  xllm::XLLM_InitLLMOptions options;
  bool ret = llm.Initialize("/export/home/models/Qwen3-8B", "auto", options);
  if (!ret) {
    std::cout << "llm init failed" << std::endl;
    return -1;
  }

  std::cout << "llm init succefully" << std::endl;

  std::string prompt = "hello, 京东";
  xllm::XLLM_RequestParams params;
  params.max_tokens = 100;
  llm.Generate("Qwen3-8B",
               prompt,
               params,
               [](const xllm::XLLM_Response& response) -> bool {
                 std::cout << "output:" << response.choices[0].text.c_str()
                           << std::endl;
                 return true;
               });

  std::cout << "llm test end" << std::endl;

  sleep(10);

  return 0;
}