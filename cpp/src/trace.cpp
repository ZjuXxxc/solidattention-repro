#include "solidattention/trace.hpp"

#include <chrono>
#include <fstream>
#include <stdexcept>

namespace solidattention {
namespace {

std::uint64_t monotonic_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::string escape(const std::string& value) {
  std::string output;
  for (const char character : value) {
    if (character == '"' || character == '\\') output.push_back('\\');
    output.push_back(character);
  }
  return output;
}

const char* lane_name(int lane) {
  switch (lane) {
    case 1: return "SSD read";
    case 2: return "PCIe H2D";
    case 3: return "GPU compute";
    case 4: return "PCIe D2H";
    default: return "DRAM staging";
  }
}

}  // namespace

Trace::Trace() : origin_ns_(monotonic_ns()) {}

std::uint64_t Trace::now_us() const {
  return (monotonic_ns() - origin_ns_) / 1000;
}

void Trace::add(TraceEvent event) { events_.push_back(std::move(event)); }

void Trace::write(const std::string& path) const {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot write trace: " + path);
  output << "{\"traceEvents\":[\n";
  for (std::size_t i = 0; i < events_.size(); ++i) {
    const auto& event = events_[i];
    output << "{\"name\":\"" << escape(event.name) << "\",\"cat\":\""
           << escape(event.category) << "\",\"ph\":\"X\",\"ts\":"
           << event.start_us << ",\"dur\":" << event.duration_us
           << ",\"pid\":1,\"tid\":\"" << lane_name(event.lane) << "\""
           << ",\"args\":{\"step\":" << event.step << ",\"layer\":"
           << event.layer << ",\"bytes\":" << event.bytes << "}}";
    if (i + 1 != events_.size()) output << ',';
    output << '\n';
  }
  output << "],\"displayTimeUnit\":\"ms\"}\n";
}

}  // namespace solidattention
