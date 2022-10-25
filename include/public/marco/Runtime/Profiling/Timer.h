#ifndef MARCO_RUNTIME_PROFILING_TIMER_H
#define MARCO_RUNTIME_PROFILING_TIMER_H

#include <chrono>

namespace marco::runtime::profiling
{
  class Timer
  {
    public:
      void start();

      void stop();

      void reset();

      double totalElapsedTime() const;

    private:
      std::chrono::nanoseconds elapsed() const;

    private:
      bool running_ = false;
      std::chrono::steady_clock::time_point start_;
      std::chrono::nanoseconds accumulatedTime_;
  };
}

#endif // MARCO_RUNTIME_PROFILING_TIMER_H
