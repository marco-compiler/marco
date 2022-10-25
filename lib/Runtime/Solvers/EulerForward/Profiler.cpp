#include "marco/Runtime/Solvers/EulerForward/Profiler.h"
#include <iostream>

#ifdef MARCO_PROFILING

namespace marco::runtime::profiling
{
  EulerForwardProfiler::EulerForwardProfiler()
      : Profiler("Euler forward")
  {
    registerProfiler(*this);
  }

  void EulerForwardProfiler::reset()
  {
    initialConditions.reset();
    nonStateVariables.reset();
    stateVariables.reset();
  }

  void EulerForwardProfiler::print() const
  {
    std::cerr << "Time spent on initial conditions computation: " << initialConditions.totalElapsedTime() << " ms\n";
    std::cerr << "Time spent on state variables computation: " << stateVariables.totalElapsedTime() << " ms\n";
    std::cerr << "Time spent on non-state variables computation: " << nonStateVariables.totalElapsedTime() << " ms\n";
  }

  EulerForwardProfiler& eulerForwardProfiler()
  {
    static EulerForwardProfiler obj;
    return obj;
  }
}

#endif // MARCO_PROFILING
