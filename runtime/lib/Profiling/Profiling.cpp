#include "marco/Runtime/Profiling/Profiling.h"
#include "marco/Runtime/Profiling/Statistics.h"

using namespace ::marco::runtime::profiling;

static Statistics& statistics()
{
  static Statistics obj;
  return obj;
}

namespace marco::runtime
{
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

  void registerProfiler(std::shared_ptr<Profiler> profiler)
  {
    ::statistics().registerProfiler(profiler);
  }
}
