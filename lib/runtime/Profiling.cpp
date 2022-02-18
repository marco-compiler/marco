#ifdef MARCO_PROFILING

#include "marco/runtime/Profiling.h"
#include <iostream>
#include <vector>

void Timer::start()
{
  start_ = std::chrono::steady_clock::now();
  running_ = true;
}

void Timer::stop()
{
  accumulatedTime_ += elapsed();
  running_ = false;
}

void Timer::reset()
{
  accumulatedTime_ = std::chrono::duration_values<std::chrono::nanoseconds>::zero();
}

double Timer::totalElapsedTime() const
{
  if (running_) {
    return static_cast<double>((accumulatedTime_ + elapsed()).count()) / 1e6;
  }

  return static_cast<double>(accumulatedTime_.count()) / 1e6;
}

std::chrono::nanoseconds Timer::elapsed() const
{
  return std::chrono::steady_clock::now() - start_;
}

Profiler::Profiler(const std::string& name) : name(name)
{
}

Profiler::Profiler(const Profiler& other) = default;

Profiler::~Profiler() = default;

const std::string& Profiler::getName() const
{
  return name;
}

namespace
{
  class Statistics
  {
    public:
      void registerProfiler(Profiler& profiler)
      {
        profilers.push_back(&profiler);
      }

      void reset()
      {
        for (const auto& profiler : profilers) {
          profiler->reset();
        }
      }

      void print() const
      {
        constexpr size_t lineWidth = 80;

        std::cerr << "\n";
        printHeader(lineWidth, "Runtime statistics");

        for (const auto& profiler : profilers) {
          printProfilerTitle(lineWidth, profiler->getName());
          profiler->print();
          std::cerr << "\n";
        }
      }

    private:
      void printHeaderLine(size_t width) const
      {
        for (size_t i = 0; i < width; ++i) {
          std::cerr << "-";
        }

        std::cerr << "\n";
      }

      void printHeaderTitle(size_t width, const std::string& title) const
      {
        size_t spaces = width - 2 - title.size();
        size_t left = spaces / 2;
        size_t right = spaces - left;

        std::cerr << "|";

        for (size_t i = 0; i < left; ++i) {
          std::cerr << " ";
        }

        std::cerr << title.data();

        for (size_t i = 0; i < right; ++i) {
          std::cerr << " ";
        }

        std::cerr << "|\n";
      }

      void printHeader(size_t width, const std::string& title) const
      {
        printHeaderLine(width);
        printHeaderTitle(width, title);
        printHeaderLine(width);
      }

      void printProfilerTitle(size_t width, const std::string& title) const
      {
        size_t symbols = width - 2 - title.size();
        size_t left = symbols / 2;
        size_t right = symbols - left;

        for (size_t i = 0; i < left; ++i) {
          std::cerr << "=";
        }

        std::cerr << " " << title.data() << " ";

        for (size_t i = 0; i < right; ++i) {
          std::cerr << "=";
        }

        std::cerr << "\n";
      }

      std::vector<Profiler*> profilers;
  };

  ::Statistics& statistics()
  {
    static ::Statistics obj;
    return obj;
  }
}

void profilingInit()
{
  ::statistics().reset();
}

void printProfilingStats()
{
  ::statistics().print();
}

void registerProfiler(Profiler& profiler)
{
  ::statistics().registerProfiler(profiler);
}

#endif
