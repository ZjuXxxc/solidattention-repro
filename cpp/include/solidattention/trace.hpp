#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace solidattention {

struct TraceEvent {
  std::string name;
  std::string category;
  std::uint64_t start_us;
  std::uint64_t duration_us;
  int lane;
  std::size_t step;
  std::size_t layer;
  std::size_t bytes;
};

class Trace {
 public:
  Trace();
  std::uint64_t now_us() const;
  void add(TraceEvent event);
  void write(const std::string& path) const;

 private:
  std::uint64_t origin_ns_;
  std::vector<TraceEvent> events_;
};

}  // namespace solidattention
