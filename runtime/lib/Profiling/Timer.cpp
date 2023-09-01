#include "marco/Runtime/Profiling/Timer.h"

using namespace ::std::chrono;

namespace marco::runtime::profiling
{
  void Timer::start()
  {
    std::lock_guard<std::mutex> lockGuard(mutex);

    if (running_++ == 0) {
      start_ = steady_clock::now();
    }
  }

  void Timer::stop()
  {
    std::lock_guard<std::mutex> lockGuard(mutex);

    if (--running_ == 0) {
      accumulatedTime_ += elapsed();
    }
  }

  void Timer::reset()
  {
    std::lock_guard<std::mutex> lockGuard(mutex);
    accumulatedTime_ = duration_values<nanoseconds>::zero();
  }

  double Timer::totalElapsedTime() const
  {
    std::lock_guard<std::mutex> lockGuard(mutex);

    if (running_) {
      return static_cast<double>((accumulatedTime_ + elapsed()).count()) / 1e6;
    }

    return static_cast<double>(accumulatedTime_.count()) / 1e6;
  }

  nanoseconds Timer::elapsed() const
  {
    return steady_clock::now() - start_;
  }
}
