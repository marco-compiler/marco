#include "marco/Runtime/Profiling/Timer.h"

using namespace ::std::chrono;

namespace marco::runtime::profiling
{
  Timer::Timer()
  {
    reset();
  }

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

  nanoseconds Timer::elapsed() const
  {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        steady_clock::now() - start_);
  }

  std::chrono::nanoseconds Timer::totalElapsed() const
  {
    std::lock_guard<std::mutex> lockGuard(mutex);

    if (running_) {
      return accumulatedTime_ + elapsed();
    }

    return accumulatedTime_;
  }
}
