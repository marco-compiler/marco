#include "marco/Runtime/Solvers/KINSOL/Profiler.h"
#include "marco/Runtime/Profiling/Profiling.h"
#include <iostream>

#ifdef MARCO_PROFILING

namespace marco::runtime::profiling
{
  KINSOLProfiler::KINSOLProfiler()
      : Profiler("KINSOL")
  {
    registerProfiler(*this);
  }

  void KINSOLProfiler::reset()
  {
    stepsTimer.reset();
  }

  void KINSOLProfiler::print() const
  {
    std::cerr << "Time spent on KINSOL steps: " << stepsTimer.totalElapsedTime() << " ms\n";
  }

  KINSOLProfiler& kinsolProfiler()
  {
    static KINSOLProfiler obj;
    return obj;
  }
}

#endif // MARCO_PROFILING
