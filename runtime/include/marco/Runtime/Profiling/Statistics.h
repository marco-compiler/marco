#ifndef MARCO_RUNTIME_PROFILING_STATISTICS_H
#define MARCO_RUNTIME_PROFILING_STATISTICS_H

#include "marco/Runtime/Profiling/Profiler.h"
#include <memory>
#include <vector>

namespace marco::runtime::profiling
{
  class Statistics
  {
    public:
      void registerProfiler(Profiler& profiler);
      void registerProfiler(std::shared_ptr<Profiler> profiler);

      void reset();

      void print() const;

    private:
      void printHeaderLine(size_t width) const;

      void printHeaderTitle(size_t width, const std::string& title) const;

      void printHeader(size_t width, const std::string& title) const;

      void printProfilerTitle(size_t width, const std::string& title) const;

    private:
      std::vector<Profiler*> profilers;
      std::vector<std::shared_ptr<Profiler>> sharedProfilers;
  };
}

#endif // MARCO_RUNTIME_PROFILING_STATISTICS_H
