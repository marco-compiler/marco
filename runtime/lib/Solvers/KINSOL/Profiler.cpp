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
    std::lock_guard<std::mutex> lockGuard(mutex);
    residualsCallCounter = 0;
    residualsTimer.reset();
    partialDerivativesCallCounter = 0;
    partialDerivativesTimer.reset();
    copyVarsFromMARCOTimer.reset();
    copyVarsIntoMARCOTimer.reset();
  }

  void KINSOLProfiler::print() const
  {
    std::lock_guard<std::mutex> lockGuard(mutex);

    std::cerr << "Number of computations of the residuals: " << residualsCallCounter << "\n";
    std::cerr << "Time spent on computing the residuals: " << residualsTimer.totalElapsedTime() << " ms\n";
    std::cerr << "Number of computations of the partial derivatives: " << partialDerivativesCallCounter << "\n";
    std::cerr << "Time spent on computing the partial derivatives: " << partialDerivativesTimer.totalElapsedTime() << " ms\n";
    std::cerr << "Time spent on copying the variables from MARCO: " << copyVarsFromMARCOTimer.totalElapsedTime() << " ms\n";
    std::cerr << "Time spent on copying the variables into MARCO: " << copyVarsIntoMARCOTimer.totalElapsedTime() << " ms\n";
  }

  KINSOLProfiler& kinsolProfiler()
  {
    static KINSOLProfiler obj;
    return obj;
  }
}

#endif // MARCO_PROFILING
