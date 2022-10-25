#ifndef MARCO_RUNTIME_PROFILING_PROFILING_H
#define MARCO_RUNTIME_PROFILING_PROFILING_H

#include "marco/Runtime/Profiling/Profiler.h"

namespace marco::runtime
{
  void registerProfiler(profiling::Profiler& profiler);

  void profilingInit();
  void printProfilingStats();
}

#endif // MARCO_RUNTIME_PROFILING_PROFILING_H
