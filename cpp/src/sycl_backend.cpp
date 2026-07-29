#include "solidattention/backend.hpp"

#include <sycl/sycl.hpp>

#include <memory>
#include <stdexcept>
#include <vector>

namespace solidattention {

class SyclBackend final : public AcceleratorBackend {
 public:
  SyclBackend()
      : queue_(sycl::default_selector_v,
               sycl::property::queue::enable_profiling{}) {}
  std::string name() const override {
    return "oneapi-sycl:" +
           queue_.get_device().get_info<sycl::info::device::name>();
  }
  void* allocate_host(std::size_t bytes) override {
    auto* pointer = sycl::malloc_host<std::uint8_t>(bytes, queue_);
    if (!pointer) throw std::bad_alloc();
    return pointer;
  }
  void free_host(void* pointer) override { sycl::free(pointer, queue_); }
  TransferResult execute(void* pinned_input, std::size_t bytes,
                         std::uint8_t mask) override {
    auto* input = sycl::malloc_device<std::uint8_t>(bytes, queue_);
    auto* output = sycl::malloc_device<std::uint8_t>(bytes, queue_);
    std::vector<std::uint8_t> host_output(bytes);
    auto h2d = queue_.memcpy(input, pinned_input, bytes);
    auto kernel = queue_.submit([&](sycl::handler& handler) {
      handler.depends_on(h2d);
      handler.parallel_for(sycl::range<1>(bytes), [=](sycl::id<1> id) {
        const auto index = id[0];
        output[index] = input[index] ^ mask;
      });
    });
    auto d2h = queue_.submit([&](sycl::handler& handler) {
      handler.depends_on(kernel);
      handler.memcpy(host_output.data(), output, bytes);
    });
    d2h.wait();
    auto elapsed = [](const sycl::event& event) {
      const auto start = event.get_profiling_info<
          sycl::info::event_profiling::command_start>();
      const auto end = event.get_profiling_info<
          sycl::info::event_profiling::command_end>();
      return static_cast<double>(end - start) / 1e6;
    };
    std::uint64_t checksum = 0;
    for (const auto value : host_output) checksum += value;
    sycl::free(input, queue_);
    sycl::free(output, queue_);
    return {.h2d_ms = elapsed(h2d), .kernel_ms = elapsed(kernel),
            .d2h_ms = elapsed(d2h), .checksum = checksum};
  }

 private:
  sycl::queue queue_;
};

std::unique_ptr<AcceleratorBackend> make_sycl_backend() {
  return std::make_unique<SyclBackend>();
}

}  // namespace solidattention
