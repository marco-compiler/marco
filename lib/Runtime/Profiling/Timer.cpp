#include "marco/Runtime/Profiling/Timer.h"

using namespace ::std::chrono;

namespace marco::runtime::profiling
{
  void Timer::start()
  {
    start_ = steady_clock::now();
    running_ = true;
  }

  void Timer::stop()
  {
    accumulatedTime_ += elapsed();
    running_ = false;
  }

  void Timer::reset()
  {
    accumulatedTime_ = duration_values<nanoseconds>::zero();
  }

  double Timer::totalElapsedTime() const
  {
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
