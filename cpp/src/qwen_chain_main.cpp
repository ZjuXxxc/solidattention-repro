#define SOLIDATTENTION_QWEN_LAYER_LIBRARY
#include "qwen_layer_main.cpp"

#include <sstream>

namespace {

struct PlanLayer {
  int layer{};
  std::string directory;
  std::uint64_t kv_offset{};
  std::vector<std::size_t> selected;
};

std::vector<PlanLayer> read_plan(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot read chain plan");
  std::vector<PlanLayer> result;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    std::stringstream row(line);
    PlanLayer entry;
    std::string offset, blocks, block;
    std::getline(row, offset, '\t');
    entry.layer = std::stoi(offset);
    std::getline(row, entry.directory, '\t');
    std::getline(row, offset, '\t');
    entry.kv_offset = std::stoull(offset);
    std::getline(row, blocks, '\t');
    std::stringstream block_stream(blocks);
    while (std::getline(block_stream, block, ',')) {
      entry.selected.push_back(std::stoul(block));
    }
    if (entry.selected.size() != 4) throw std::runtime_error("plan block count");
    result.push_back(std::move(entry));
  }
  return result;
}

struct LayerAudit {
  int layer{};
  double weight_read_ms{};
  double weight_h2d_ms{};
  double kv_read_ms{};
  double kv_h2d_ms{};
  double exposed_prefetch_wait_ms{};
  double compute_ms{};
  Error output_error{};
  Error input_error{};
};

struct ResidentWeights {
  std::unique_ptr<Device> input_norm, q_weight, q_norm, o_weight, post_norm;
  std::unique_ptr<Device> gate_weight, up_weight, down_weight;
  std::size_t bytes{};
};

std::unique_ptr<Device> resident_tensor(const std::vector<float>& host,
                                        std::size_t* bytes) {
  auto device = std::make_unique<Device>(host.size() * sizeof(float));
  upload(*device, host);
  *bytes += host.size() * sizeof(float);
  return device;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path fixture = "artifacts/qwen-28-layers";
    std::filesystem::path plan_path;
    std::filesystem::path metrics = "artifacts/cpp-p1-2f-metrics.json";
    bool resident_weights = false;
    bool pipeline_kv = false;
    bool final_audit_only = false;
    bool dram_prefetch_all = false;
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--fixture" && index + 1 < argc) fixture = argv[++index];
      else if (argument == "--plan" && index + 1 < argc) plan_path = argv[++index];
      else if (argument == "--metrics" && index + 1 < argc) metrics = argv[++index];
      else if (argument == "--resident-weights") resident_weights = true;
      else if (argument == "--pipeline-kv") pipeline_kv = true;
      else if (argument == "--final-audit-only") final_audit_only = true;
      else if (argument == "--dram-prefetch-all") dram_prefetch_all = true;
      else throw std::runtime_error("unknown chain argument: " + argument);
    }
    if (pipeline_kv && !resident_weights) {
      throw std::runtime_error("--pipeline-kv requires --resident-weights");
    }
    if (dram_prefetch_all && !pipeline_kv) {
      throw std::runtime_error("--dram-prefetch-all requires --pipeline-kv");
    }
    const auto plan = read_plan(plan_path);
    if (plan.empty()) throw std::runtime_error("empty chain plan");
    constexpr int hidden = 1024, intermediate = 3072;
    constexpr int q_heads = 16, kv_heads = 8, head_dim = 128;
    constexpr int q_width = q_heads * head_dim, kv_width = kv_heads * head_dim;
    constexpr float epsilon = 1e-6f, theta = 1000000.0f;
    constexpr std::size_t block_bytes = 128 * 1024;
    constexpr std::size_t packed_bytes = 4 * block_bytes;

    cudaStream_t stream{};
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "chain stream");
    cublasHandle_t handle{};
    cublas_check(cublasCreate_v2(&handle), "chain cublasCreate");
    cublas_check(cublasSetStream_v2(handle, stream), "chain cublasSetStream");
    Kernels kernels;
    const float one = 1.0f, zero = 0.0f;

    Device d_hidden(hidden * 4), d_normalized(hidden * 4);
    Device d_q(q_width * 4), d_attended(q_width * 4);
    Device d_attention_output(hidden * 4), d_attention_residual(hidden * 4);
    Device d_post_normalized(hidden * 4), d_gate(intermediate * 4);
    Device d_up(intermediate * 4), d_activated(intermediate * 4);
    Device d_mlp_output(hidden * 4), d_output(hidden * 4);
    Device d_kv0(packed_bytes), d_kv1(packed_bytes);
    Device d_input_norm(hidden * 4), d_q_weight(q_width * hidden * 4);
    Device d_q_norm(head_dim * 4), d_o_weight(hidden * q_width * 4);
    Device d_post_norm(hidden * 4), d_gate_weight(intermediate * hidden * 4);
    Device d_up_weight(intermediate * hidden * 4);
    Device d_down_weight(hidden * intermediate * 4);

    const std::size_t host_slot_count = dram_prefetch_all ? plan.size() : 2;
    std::vector<void*> pinned(host_slot_count, nullptr);
    for (std::size_t slot = 0; slot < host_slot_count; ++slot) {
      cuda_check(cudaHostAlloc(&pinned[slot], packed_bytes, cudaHostAllocDefault),
                 "chain pinned KV");
    }
    solidattention::UringReader reader(
        (fixture / "all-layers-kv-fp16.bin").string(), pinned,
        packed_bytes);
    cudaStream_t copy_stream{};
    cuda_check(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking),
               "chain copy stream");
    std::vector<double> staged_read_ms(plan.size(), 0.0);
    std::vector<double> staged_h2d_ms(plan.size(), 0.0);
    double dram_prefetch_ms = 0.0;
    if (pipeline_kv) {
      const auto dram_begin = std::chrono::steady_clock::now();
      const std::size_t preload_layers = dram_prefetch_all ? plan.size() : 1;
      for (std::size_t layer = 0; layer < preload_layers; ++layer) {
        std::vector<std::uint64_t> offsets;
        for (const auto block : plan[layer].selected) {
          offsets.push_back(plan[layer].kv_offset + block * block_bytes);
        }
        staged_read_ms[layer] = reader.read_blocks_fixed(
            dram_prefetch_all ? layer : 0, offsets, block_bytes);
      }
      dram_prefetch_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - dram_begin).count();
      const auto begin = std::chrono::steady_clock::now();
      cudaMemcpyAsync(d_kv0.pointer, pinned[0], packed_bytes,
                      cudaMemcpyHostToDevice, copy_stream);
      cudaStreamSynchronize(copy_stream);
      staged_h2d_ms[0] = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - begin).count();
    }

    const auto initial = load(fixture / plan[0].directory, "chain_input", hidden);
    upload(d_hidden, initial);
    std::vector<ResidentWeights> resident;
    double resident_preload_ms = 0.0;
    std::size_t resident_weight_bytes = 0;
    if (resident_weights) {
      resident.reserve(plan.size());
      const auto preload_begin = std::chrono::steady_clock::now();
      for (const auto& entry : plan) {
        const auto directory = fixture / entry.directory;
        ResidentWeights weights;
        weights.input_norm = resident_tensor(
            load(directory, "input_norm_weight", hidden), &weights.bytes);
        weights.q_weight = resident_tensor(
            load(directory, "q_weight", q_width * hidden), &weights.bytes);
        weights.q_norm = resident_tensor(
            load(directory, "q_norm_weight", head_dim), &weights.bytes);
        weights.o_weight = resident_tensor(
            load(directory, "o_weight", hidden * q_width), &weights.bytes);
        weights.post_norm = resident_tensor(
            load(directory, "post_norm_weight", hidden), &weights.bytes);
        weights.gate_weight = resident_tensor(
            load(directory, "gate_weight", intermediate * hidden), &weights.bytes);
        weights.up_weight = resident_tensor(
            load(directory, "up_weight", intermediate * hidden), &weights.bytes);
        weights.down_weight = resident_tensor(
            load(directory, "down_weight", hidden * intermediate), &weights.bytes);
        resident_weight_bytes += weights.bytes;
        resident.push_back(std::move(weights));
      }
      cudaStreamSynchronize(stream);
      resident_preload_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - preload_begin).count();
    }
    std::vector<LayerAudit> audits;
    const auto total_begin = std::chrono::steady_clock::now();
    for (std::size_t plan_index = 0; plan_index < plan.size(); ++plan_index) {
      const auto& entry = plan[plan_index];
      const auto directory = fixture / entry.directory;
      LayerAudit audit;
      audit.layer = entry.layer;
      float *p_input_norm, *p_q_weight, *p_q_norm, *p_o_weight, *p_post_norm;
      float *p_gate_weight, *p_up_weight, *p_down_weight;
      if (resident_weights) {
        auto& weights = resident[plan_index];
        p_input_norm = weights.input_norm->f32();
        p_q_weight = weights.q_weight->f32();
        p_q_norm = weights.q_norm->f32();
        p_o_weight = weights.o_weight->f32();
        p_post_norm = weights.post_norm->f32();
        p_gate_weight = weights.gate_weight->f32();
        p_up_weight = weights.up_weight->f32();
        p_down_weight = weights.down_weight->f32();
      } else {
        const auto weight_begin = std::chrono::steady_clock::now();
        const auto input_norm = load(directory, "input_norm_weight", hidden);
        const auto q_weight = load(directory, "q_weight", q_width * hidden);
        const auto q_norm = load(directory, "q_norm_weight", head_dim);
        const auto o_weight = load(directory, "o_weight", hidden * q_width);
        const auto post_norm = load(directory, "post_norm_weight", hidden);
        const auto gate_weight = load(directory, "gate_weight", intermediate * hidden);
        const auto up_weight = load(directory, "up_weight", intermediate * hidden);
        const auto down_weight = load(directory, "down_weight", hidden * intermediate);
        audit.weight_read_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - weight_begin).count();
        const auto upload_begin = std::chrono::steady_clock::now();
        upload(d_input_norm, input_norm); upload(d_q_weight, q_weight);
        upload(d_q_norm, q_norm); upload(d_o_weight, o_weight);
        upload(d_post_norm, post_norm); upload(d_gate_weight, gate_weight);
        upload(d_up_weight, up_weight); upload(d_down_weight, down_weight);
        audit.weight_h2d_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - upload_begin).count();
        p_input_norm = d_input_norm.f32(); p_q_weight = d_q_weight.f32();
        p_q_norm = d_q_norm.f32(); p_o_weight = d_o_weight.f32();
        p_post_norm = d_post_norm.f32(); p_gate_weight = d_gate_weight.f32();
        p_up_weight = d_up_weight.f32(); p_down_weight = d_down_weight.f32();
      }
      const std::size_t current_slot = plan_index % 2;
      void* current_kv = current_slot == 0 ? d_kv0.pointer : d_kv1.pointer;
      if (pipeline_kv) {
        audit.kv_read_ms = staged_read_ms[plan_index];
        audit.kv_h2d_ms = staged_h2d_ms[plan_index];
      } else {
        std::vector<std::uint64_t> offsets;
        for (const auto block : entry.selected) {
          offsets.push_back(entry.kv_offset + block * block_bytes);
        }
        audit.kv_read_ms = reader.read_blocks_fixed(0, offsets, block_bytes);
        const auto copy_begin = std::chrono::steady_clock::now();
        cudaMemcpyAsync(current_kv, pinned[0], packed_bytes,
                        cudaMemcpyHostToDevice, copy_stream);
        cudaStreamSynchronize(copy_stream);
        audit.kv_h2d_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - copy_begin).count();
      }
      if (pipeline_kv && !dram_prefetch_all && plan_index + 1 < plan.size()) {
        const auto& next = plan[plan_index + 1];
        std::vector<std::uint64_t> next_offsets;
        for (const auto block : next.selected) {
          next_offsets.push_back(next.kv_offset + block * block_bytes);
        }
        reader.submit_blocks_fixed((plan_index + 1) % 2, next_offsets,
                                   block_bytes);
      }

      const auto compute_begin = std::chrono::steady_clock::now();
      kernels.rms(d_hidden.f32(), p_input_norm, d_normalized.f32(),
                  hidden, epsilon, stream);
      cublas_check(cublasSgemm_v2(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, q_width, 1, hidden, &one,
          p_q_weight, hidden, d_normalized.f32(), hidden, &zero,
          d_q.f32(), q_width), "chain Q projection");
      kernels.head_norm(d_q.f32(), p_q_norm, q_heads, head_dim,
                        epsilon, stream);
      kernels.apply_rope(d_q.f32(), 1, q_heads, head_dim, 511, theta, stream);
      kernels.sparse_attention(d_q.f32(), current_kv, d_attended.f32(), 128,
                               q_heads, kv_heads, head_dim, stream);
      cublas_check(cublasSgemm_v2(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, hidden, 1, q_width, &one,
          p_o_weight, q_width, d_attended.f32(), q_width, &zero,
          d_attention_output.f32(), hidden), "chain O projection");
      kernels.add(d_hidden.f32(), d_attention_output.f32(),
                  d_attention_residual.f32(), hidden, stream);
      kernels.rms(d_attention_residual.f32(), p_post_norm,
                  d_post_normalized.f32(), hidden, epsilon, stream);
      cudaEvent_t next_copy_begin{}, next_copy_end{};
      if (pipeline_kv && plan_index + 1 < plan.size()) {
        const auto wait_begin = std::chrono::steady_clock::now();
        if (!dram_prefetch_all) {
          staged_read_ms[plan_index + 1] = reader.wait_blocks_fixed();
          audit.exposed_prefetch_wait_ms =
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - wait_begin).count();
        }
        const std::size_t next_slot = (plan_index + 1) % 2;
        void* next_host = pinned[dram_prefetch_all ? plan_index + 1 : next_slot];
        void* next_device = next_slot == 0 ? d_kv0.pointer : d_kv1.pointer;
        cudaEventCreate(&next_copy_begin); cudaEventCreate(&next_copy_end);
        cudaEventRecord(next_copy_begin, copy_stream);
        cudaMemcpyAsync(next_device, next_host, packed_bytes,
                        cudaMemcpyHostToDevice, copy_stream);
        cudaEventRecord(next_copy_end, copy_stream);
      }
      auto mlp = [&](float* weight, float* output) {
        cublas_check(cublasSgemm_v2(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, intermediate, 1, hidden, &one,
            weight, hidden, d_post_normalized.f32(), hidden, &zero, output,
            intermediate), "chain MLP projection");
      };
      mlp(p_gate_weight, d_gate.f32());
      mlp(p_up_weight, d_up.f32());
      kernels.silu(d_gate.f32(), d_up.f32(), d_activated.f32(), intermediate,
                   stream);
      cublas_check(cublasSgemm_v2(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, hidden, 1, intermediate, &one,
          p_down_weight, intermediate, d_activated.f32(), intermediate,
          &zero, d_mlp_output.f32(), hidden), "chain down projection");
      kernels.add(d_attention_residual.f32(), d_mlp_output.f32(),
                  d_output.f32(), hidden, stream);
      cudaStreamSynchronize(stream);
      if (next_copy_end) {
        cudaEventSynchronize(next_copy_end);
        float next_copy_ms = 0.0f;
        cudaEventElapsedTime(&next_copy_ms, next_copy_begin, next_copy_end);
        staged_h2d_ms[plan_index + 1] = next_copy_ms;
        cudaEventDestroy(next_copy_begin); cudaEventDestroy(next_copy_end);
      }
      audit.compute_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - compute_begin).count();
      if (!final_audit_only) {
        const auto actual = download(d_output.f32(), hidden);
        audit.output_error = compare(
            actual, load(directory, "chain_sparse_layer_output", hidden));
        audit.input_error = compare(
            download(d_hidden.f32(), hidden), load(directory, "chain_input", hidden));
      } else {
        audit.output_error = {-1.0, -1.0};
        audit.input_error = {-1.0, -1.0};
      }
      cudaMemcpyAsync(d_hidden.pointer, d_output.pointer, hidden * 4,
                      cudaMemcpyDeviceToDevice, stream);
      if (!final_audit_only) cudaStreamSynchronize(stream);
      audits.push_back(audit);
      if (!final_audit_only) {
        std::cout << "layer=" << entry.layer
                  << " output_error=" << audit.output_error.maximum
                  << " cosine=" << audit.output_error.cosine << '\n';
      }
    }
    cudaStreamSynchronize(stream);
    const double total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - total_begin).count();
    const auto final_error = compare(
        download(d_hidden.f32(), hidden),
        load(fixture / plan.back().directory, "chain_sparse_layer_output", hidden));
    std::filesystem::create_directories(metrics.parent_path());
    std::ofstream report(metrics);
    report << std::fixed << std::setprecision(9)
           << "{\n  \"version\": \""
           << (resident_weights && pipeline_kv && final_audit_only && dram_prefetch_all
                   ? "P1.2g.3-pinned-dram-upper-bound"
                   : resident_weights && pipeline_kv && final_audit_only
                   ? "P1.2g.2-production-timing-boundary"
                   : resident_weights && pipeline_kv
                   ? "P1.2g.1-resident-fp32-kv-pipeline"
                   : resident_weights ? "P1.2g-resident-fp32-qwen-chain"
                                : "P1.2f-single-process-qwen-chain")
           << "\",\n"
           << "  \"layers\": " << audits.size() << ",\n"
           << "  \"resident_weights\": " << (resident_weights ? "true" : "false") << ",\n"
           << "  \"resident_preload_ms\": " << resident_preload_ms << ",\n"
           << "  \"resident_weight_bytes\": " << resident_weight_bytes << ",\n"
           << "  \"pipeline_kv\": " << (pipeline_kv ? "true" : "false") << ",\n"
           << "  \"final_audit_only\": " << (final_audit_only ? "true" : "false") << ",\n"
           << "  \"dram_prefetch_all\": " << (dram_prefetch_all ? "true" : "false") << ",\n"
           << "  \"dram_prefetch_ms\": " << dram_prefetch_ms << ",\n"
           << "  \"pinned_dram_bytes\": " << host_slot_count * packed_bytes << ",\n"
           << "  \"total_wall_ms\": " << total_ms << ",\n"
           << "  \"final_output_max_error\": " << final_error.maximum << ",\n"
           << "  \"final_output_cosine\": " << final_error.cosine << ",\n"
           << "  \"layers_detail\": [\n";
    for (std::size_t index = 0; index < audits.size(); ++index) {
      const auto& audit = audits[index];
      report << "    {\"layer\": " << audit.layer
             << ", \"weight_read_ms\": " << audit.weight_read_ms
             << ", \"weight_h2d_ms\": " << audit.weight_h2d_ms
             << ", \"kv_read_ms\": " << audit.kv_read_ms
             << ", \"kv_h2d_ms\": " << audit.kv_h2d_ms
             << ", \"exposed_prefetch_wait_ms\": "
             << audit.exposed_prefetch_wait_ms
             << ", \"compute_ms\": " << audit.compute_ms
             << ", \"input_max_error\": " << audit.input_error.maximum
             << ", \"input_cosine\": " << audit.input_error.cosine
             << ", \"output_max_error\": " << audit.output_error.maximum
             << ", \"output_cosine\": " << audit.output_error.cosine << "}";
      report << (index + 1 == audits.size() ? "\n" : ",\n");
    }
    report << "  ]\n}\n";
    for (void* buffer : pinned) cudaFreeHost(buffer);
    cublasDestroy_v2(handle);
    cudaStreamDestroy(stream);
    cudaStreamDestroy(copy_stream);
    std::cout << "total_wall_ms=" << total_ms << "\nmetrics=" << metrics << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
