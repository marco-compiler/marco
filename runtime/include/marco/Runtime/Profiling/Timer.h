#ifndef MARCO_RUNTIME_PROFILING_TIMER_H
#define MARCO_RUNTIME_PROFILING_TIMER_H

#include <chrono>
#include <mutex>

namespace marco::runtime::profiling
{
  class Timer
  {
    public:
      Timer();

      Timer(const Timer& other) = delete;

      Timer& operator=(const Timer& other) = delete;

      void start();

      void stop();

      void reset();

      double totalElapsedTime() const;

    private:
      std::chrono::nanoseconds elapsed() const;

    private:
      int running_ = 0;
      std::chrono::steady_clock::time_point start_;
      std::chrono::nanoseconds accumulatedTime_;
      mutable std::mutex mutex;
  };
}

#endif // MARCO_RUNTIME_PROFILING_TIMER_H
