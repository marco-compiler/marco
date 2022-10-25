#ifndef MARCO_RUNTIME_PROFILING_PROFILER_H
#define MARCO_RUNTIME_PROFILING_PROFILER_H

#include <string>

namespace marco::runtime::profiling
{
  class Profiler
  {
    public:
      Profiler(const std::string& name);

      Profiler(const Profiler& other) = delete;

      virtual ~Profiler();

      const std::string& getName() const;

      virtual void reset() = 0;
      virtual void print() const = 0;

    private:
      std::string name;
  };
}

#endif // MARCO_RUNTIME_PROFILING_PROFILER_H
