#include "marco/Runtime/Solvers/IDA/Profiler.h"
#include "marco/Runtime/Profiling/Profiling.h"
#include <iostream>

#ifdef MARCO_PROFILING

namespace marco::runtime::profiling
{
  IDAProfiler::IDAProfiler()
      : Profiler("IDA")
  {
    registerProfiler(*this);
  }

  void IDAProfiler::reset()
  {
    initialConditionsTimer.reset();
    stepsTimer.reset();
    algebraicVariablesTimer.reset();
  }

  void IDAProfiler::print() const
  {
    std::cerr << "Time spent on computing the initial conditions: " << initialConditionsTimer.totalElapsedTime() << " ms\n";
    std::cerr << "Time spent on IDA steps: " << stepsTimer.totalElapsedTime() << " ms\n";
    std::cerr << "Time spent on computing the algebraic variables: " << algebraicVariablesTimer.totalElapsedTime() << " ms\n";
  }

  IDAProfiler& idaProfiler()
  {
    static IDAProfiler obj;
    return obj;
  }
}

#endif // MARCO_PROFILING
