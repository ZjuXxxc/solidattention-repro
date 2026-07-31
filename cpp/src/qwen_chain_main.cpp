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
  double compute_ms{};
  Error output_error{};
  Error input_error{};
};

}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path fixture = "artifacts/qwen-28-layers";
    std::filesystem::path plan_path;
    std::filesystem::path metrics = "artifacts/cpp-p1-2f-metrics.json";
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--fixture" && index + 1 < argc) fixture = argv[++index];
      else if (argument == "--plan" && index + 1 < argc) plan_path = argv[++index];
      else if (argument == "--metrics" && index + 1 < argc) metrics = argv[++index];
      else throw std::runtime_error("unknown chain argument: " + argument);
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
    Device d_kv(packed_bytes);
    Device d_input_norm(hidden * 4), d_q_weight(q_width * hidden * 4);
    Device d_q_norm(head_dim * 4), d_o_weight(hidden * q_width * 4);
    Device d_post_norm(hidden * 4), d_gate_weight(intermediate * hidden * 4);
    Device d_up_weight(intermediate * hidden * 4);
    Device d_down_weight(hidden * intermediate * 4);

    void* pinned = nullptr;
    cuda_check(cudaHostAlloc(&pinned, packed_bytes, cudaHostAllocDefault),
               "chain pinned KV");
    solidattention::UringReader reader(
        (fixture / "all-layers-kv-fp16.bin").string(), {pinned}, packed_bytes);

    const auto initial = load(fixture / plan[0].directory, "chain_input", hidden);
    upload(d_hidden, initial);
    std::vector<LayerAudit> audits;
    const auto total_begin = std::chrono::steady_clock::now();
    for (const auto& entry : plan) {
      const auto directory = fixture / entry.directory;
      const auto weight_begin = std::chrono::steady_clock::now();
      const auto input_norm = load(directory, "input_norm_weight", hidden);
      const auto q_weight = load(directory, "q_weight", q_width * hidden);
      const auto q_norm = load(directory, "q_norm_weight", head_dim);
      const auto o_weight = load(directory, "o_weight", hidden * q_width);
      const auto post_norm = load(directory, "post_norm_weight", hidden);
      const auto gate_weight = load(directory, "gate_weight", intermediate * hidden);
      const auto up_weight = load(directory, "up_weight", intermediate * hidden);
      const auto down_weight = load(directory, "down_weight", hidden * intermediate);
      LayerAudit audit;
      audit.layer = entry.layer;
      audit.weight_read_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - weight_begin).count();
      const auto upload_begin = std::chrono::steady_clock::now();
      upload(d_input_norm, input_norm); upload(d_q_weight, q_weight);
      upload(d_q_norm, q_norm); upload(d_o_weight, o_weight);
      upload(d_post_norm, post_norm); upload(d_gate_weight, gate_weight);
      upload(d_up_weight, up_weight); upload(d_down_weight, down_weight);
      audit.weight_h2d_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - upload_begin).count();
      std::vector<std::uint64_t> offsets;
      for (const auto block : entry.selected) {
        offsets.push_back(entry.kv_offset + block * block_bytes);
      }
      audit.kv_read_ms = reader.read_blocks_fixed(0, offsets, block_bytes);
      cudaEvent_t copy_begin{}, copy_end{};
      cudaEventCreate(&copy_begin); cudaEventCreate(&copy_end);
      cudaEventRecord(copy_begin, stream);
      cudaMemcpyAsync(d_kv.pointer, pinned, packed_bytes,
                      cudaMemcpyHostToDevice, stream);
      cudaEventRecord(copy_end, stream); cudaEventSynchronize(copy_end);
      float copy_ms = 0.0f;
      cudaEventElapsedTime(&copy_ms, copy_begin, copy_end);
      audit.kv_h2d_ms = copy_ms;
      cudaEventDestroy(copy_begin); cudaEventDestroy(copy_end);

      const auto compute_begin = std::chrono::steady_clock::now();
      kernels.rms(d_hidden.f32(), d_input_norm.f32(), d_normalized.f32(),
                  hidden, epsilon, stream);
      cublas_check(cublasSgemm_v2(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, q_width, 1, hidden, &one,
          d_q_weight.f32(), hidden, d_normalized.f32(), hidden, &zero,
          d_q.f32(), q_width), "chain Q projection");
      kernels.head_norm(d_q.f32(), d_q_norm.f32(), q_heads, head_dim,
                        epsilon, stream);
      kernels.apply_rope(d_q.f32(), 1, q_heads, head_dim, 511, theta, stream);
      kernels.sparse_attention(d_q.f32(), d_kv.pointer, d_attended.f32(), 128,
                               q_heads, kv_heads, head_dim, stream);
      cublas_check(cublasSgemm_v2(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, hidden, 1, q_width, &one,
          d_o_weight.f32(), q_width, d_attended.f32(), q_width, &zero,
          d_attention_output.f32(), hidden), "chain O projection");
      kernels.add(d_hidden.f32(), d_attention_output.f32(),
                  d_attention_residual.f32(), hidden, stream);
      kernels.rms(d_attention_residual.f32(), d_post_norm.f32(),
                  d_post_normalized.f32(), hidden, epsilon, stream);
      auto mlp = [&](float* weight, float* output) {
        cublas_check(cublasSgemm_v2(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, intermediate, 1, hidden, &one,
            weight, hidden, d_post_normalized.f32(), hidden, &zero, output,
            intermediate), "chain MLP projection");
      };
      mlp(d_gate_weight.f32(), d_gate.f32());
      mlp(d_up_weight.f32(), d_up.f32());
      kernels.silu(d_gate.f32(), d_up.f32(), d_activated.f32(), intermediate,
                   stream);
      cublas_check(cublasSgemm_v2(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, hidden, 1, intermediate, &one,
          d_down_weight.f32(), intermediate, d_activated.f32(), intermediate,
          &zero, d_mlp_output.f32(), hidden), "chain down projection");
      kernels.add(d_attention_residual.f32(), d_mlp_output.f32(),
                  d_output.f32(), hidden, stream);
      cudaStreamSynchronize(stream);
      audit.compute_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - compute_begin).count();
      const auto actual = download(d_output.f32(), hidden);
      audit.output_error = compare(
          actual, load(directory, "chain_sparse_layer_output", hidden));
      audit.input_error = compare(
          download(d_hidden.f32(), hidden), load(directory, "chain_input", hidden));
      cudaMemcpyAsync(d_hidden.pointer, d_output.pointer, hidden * 4,
                      cudaMemcpyDeviceToDevice, stream);
      cudaStreamSynchronize(stream);
      audits.push_back(audit);
      std::cout << "layer=" << entry.layer
                << " output_error=" << audit.output_error.maximum
                << " cosine=" << audit.output_error.cosine << '\n';
    }
    const double total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - total_begin).count();
    std::filesystem::create_directories(metrics.parent_path());
    std::ofstream report(metrics);
    report << std::fixed << std::setprecision(9)
           << "{\n  \"version\": \"P1.2f-single-process-qwen-chain\",\n"
           << "  \"layers\": " << audits.size() << ",\n"
           << "  \"total_wall_ms\": " << total_ms << ",\n"
           << "  \"layers_detail\": [\n";
    for (std::size_t index = 0; index < audits.size(); ++index) {
      const auto& audit = audits[index];
      report << "    {\"layer\": " << audit.layer
             << ", \"weight_read_ms\": " << audit.weight_read_ms
             << ", \"weight_h2d_ms\": " << audit.weight_h2d_ms
             << ", \"kv_read_ms\": " << audit.kv_read_ms
             << ", \"kv_h2d_ms\": " << audit.kv_h2d_ms
             << ", \"compute_ms\": " << audit.compute_ms
             << ", \"input_max_error\": " << audit.input_error.maximum
             << ", \"input_cosine\": " << audit.input_error.cosine
             << ", \"output_max_error\": " << audit.output_error.maximum
             << ", \"output_cosine\": " << audit.output_error.cosine << "}";
      report << (index + 1 == audits.size() ? "\n" : ",\n");
    }
    report << "  ]\n}\n";
    cudaFreeHost(pinned);
    cublasDestroy_v2(handle);
    cudaStreamDestroy(stream);
    std::cout << "total_wall_ms=" << total_ms << "\nmetrics=" << metrics << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
