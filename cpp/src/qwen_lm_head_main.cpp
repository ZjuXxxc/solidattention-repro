#define SOLIDATTENTION_QWEN_LAYER_LIBRARY
#include "qwen_layer_main.cpp"

#include <iomanip>

int main(int argc, char** argv) {
  try {
    std::filesystem::path fixture, hidden_path, metrics, embedding_output;
    int embedding_token = -1;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--fixture" && i + 1 < argc) fixture = argv[++i];
      else if (arg == "--hidden" && i + 1 < argc) hidden_path = argv[++i];
      else if (arg == "--metrics" && i + 1 < argc) metrics = argv[++i];
      else if (arg == "--embedding-token" && i + 1 < argc) embedding_token = std::stoi(argv[++i]);
      else if (arg == "--embedding-output" && i + 1 < argc) embedding_output = argv[++i];
      else throw std::runtime_error("unknown LM-head argument");
    }
    constexpr int hidden = 1024, vocab = 151936;
    if (embedding_token >= 0) {
      if (embedding_token >= vocab || embedding_output.empty())
        throw std::runtime_error("invalid embedding request");
      auto embedding = load(fixture, "embedding",
                            static_cast<std::size_t>(vocab) * hidden);
      Device table(embedding.size() * 4), result(hidden * 4);
      upload(table, embedding);
      cudaMemcpy(result.pointer,
                 table.f32() + static_cast<std::size_t>(embedding_token) * hidden,
                 hidden * 4, cudaMemcpyDeviceToDevice);
      auto row = download(result.f32(), hidden);
      std::ofstream output(embedding_output, std::ios::binary);
      output.write(reinterpret_cast<const char*>(row.data()), hidden * 4);
      if (!output) throw std::runtime_error("cannot write embedding output");
      std::cout << "embedding_token=" << embedding_token << '\n';
      return 0;
    }
    auto x = load(hidden_path.parent_path(), hidden_path.stem().string(), hidden);
    auto norm = load(fixture, "final_norm", hidden);
    auto weight = load(fixture, "lm_head", static_cast<std::size_t>(vocab) * hidden);
    Device dx(hidden * 4), dn(hidden * 4), dnorm(hidden * 4);
    Device dw(weight.size() * 4), dlogits(static_cast<std::size_t>(vocab) * 4);
    upload(dx, x); upload(dn, norm); upload(dw, weight);
    cudaStream_t stream{}; cuda_check(cudaStreamCreate(&stream), "LM stream");
    cublasHandle_t handle{}; cublas_check(cublasCreate_v2(&handle), "LM cublas");
    cublasSetStream_v2(handle, stream); Kernels kernels;
    const float one = 1.0f, zero = 0.0f;
    const auto begin = std::chrono::steady_clock::now();
    kernels.rms(dx.f32(), dn.f32(), dnorm.f32(), hidden, 1e-6f, stream);
    cublas_check(cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_N, vocab, 1,
        hidden, &one, dw.f32(), hidden, dnorm.f32(), hidden, &zero,
        dlogits.f32(), vocab), "LM head SGEMM");
    cudaStreamSynchronize(stream);
    const double wall = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    auto logits = download(dlogits.f32(), vocab);
    auto best = std::max_element(logits.begin(), logits.end());
    std::ofstream raw(fixture / "native_logits.f32", std::ios::binary);
    raw.write(reinterpret_cast<const char*>(logits.data()), logits.size() * 4);
    std::ofstream report(metrics);
    report << std::fixed << std::setprecision(9)
      << "{\n  \"version\": \"P1.3c.0-native-lm-head\",\n"
      << "  \"vocab\": " << vocab << ",\n  \"argmax_token\": "
      << std::distance(logits.begin(), best) << ",\n  \"maximum_logit\": "
      << *best << ",\n  \"native_wall_ms\": " << wall << "\n}\n";
    cublasDestroy_v2(handle); cudaStreamDestroy(stream);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n'; return 1;
  }
}
