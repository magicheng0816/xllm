#include "llm.h"

#include "internal.h"

namespace xllm {
LLM::LLM() : initialized_(false), llm_core_(nullptr) {}

LLM::~LLM() {
  delete llm_core_->master;
  delete llm_core_;
  llm_core_ = nullptr;
}

bool LLM::Initialize(const std::string& model_path,
                     const std::string& devices,
                     const XLLM_InitLLMOptions& init_options) {
  if (!std::filesystem::exists(model_path)) {
    LOG(ERROR) << "model path[" << model_path << "] does not exist";
    return false;
  }

  std::filesystem::path model_path_fs =
      std::filesystem::path(model_path).lexically_normal();
  if (model_path_fs.has_filename()) {
    model_ids_.emplace_back(std::filesystem::path(model_path).filename());
  } else {
    model_ids_.emplace_back(
        std::filesystem::path(model_path).parent_path().filename());
  }

  xllm::Options options;
  options.model_path(model_path)
      .task_type(init_options.task)
      .devices(devices)
      .draft_model_path(init_options.draft_model)
      .draft_devices(init_options.draft_devices)
      .backend("llm")
      .block_size(init_options.block_size)
      .max_cache_size(init_options.max_cache_size)
      .max_memory_utilization(init_options.max_memory_utilization)
      .enable_prefix_cache(!init_options.disable_prefix_cache)
      .max_tokens_per_batch(init_options.max_tokens_per_batch)
      .max_seqs_per_batch(init_options.max_seqs_per_batch)
      .max_tokens_per_chunk_for_prefill(
          init_options.max_tokens_per_chunk_for_prefill)
      .num_speculative_tokens(init_options.num_speculative_tokens)
      .num_request_handling_threads(init_options.num_request_handling_threads)
      .communication_backend(init_options.communication_backend)
      .rank_tablefile(init_options.rank_tablefile)
      .expert_parallel_degree(init_options.expert_parallel_degree)
      .enable_mla(init_options.enable_mla)
      .enable_chunked_prefill(!init_options.disable_chunked_prefill)
      .master_node_addr(init_options.master_node_addr)
      .device_ip(init_options.device_ip)
      .transfer_listen_port(init_options.transfer_listen_port)
      .nnodes(init_options.nnodes)
      .node_rank(init_options.node_rank)
      .dp_size(init_options.dp_size)
      .ep_size(init_options.ep_size)
      .xservice_addr(init_options.xservice_addr)
      .instance_name(init_options.instance_name)
      .enable_disagg_pd(init_options.enable_disagg_pd)
      .enable_schedule_overlap(init_options.enable_schedule_overlap)
      .enable_pd_ooc(init_options.enable_pd_ooc)
      .kv_cache_transfer_mode(init_options.kv_cache_transfer_mode)
      .disable_ttft_profiling(init_options.disable_ttft_profiling)
      .enable_forward_interruption(init_options.enable_forward_interruption)
      .enable_shm(init_options.enable_shm)
      .is_local(init_options.is_local);

  llm_core_ = new LLMCore();
  llm_core_->master = new xllm::LLMMaster(options);
  llm_core_->master->run();

  initialized_ = true;

  return true;
}

void LLM::Generate(const std::string& model_id,
                   const std::string& prompt,
                   const XLLM_RequestParams& request_params,
                   XLLM_OutputCallback callback) {
  if (!initialized_) {
    LOG(FATAL) << "LLM is not initialized";
  }

  xllm::RequestParams xllm_request_params =
      xllm::utils::transfer_request_params(request_params);

  std::string request_id = xllm_request_params.request_id;
  int64_t created_time = absl::ToUnixSeconds(absl::Now());

  llm_core_->master->handle_request(
      prompt,
      std::nullopt,
      xllm_request_params,
      std::nullopt,
      [model_id, request_id, created_time, callback](
          const xllm::RequestOutput& req_output) -> bool {
        XLLM_Response xllm_response = xllm::utils::build_xllm_response(
            req_output, request_id, created_time, model_id);
        return callback(xllm_response);
      });
}
}  // namespace xllm