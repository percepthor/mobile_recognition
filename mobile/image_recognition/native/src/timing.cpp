#include <cstdint>
#include <chrono>

extern "C" {

uint64_t get_time_ms() {
    auto now = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
    return static_cast<uint64_t>(ms.count());
}

}  // extern "C"
