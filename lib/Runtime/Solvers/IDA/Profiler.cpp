#include "marco/Runtime/Solvers/IDA/Profiler.h"
#include "marco/Runtime/Profiling/Profiling.h"
#include <iostream>

#ifdef MARCO_PROFILING

namespace marco::runtime::profiling
{
  IDAProfiler::IDAProfiler()
      : Profiler("IDA"),
        stepsCounter(0)
  {
    registerProfiler(*this);
  }

  void IDAProfiler::reset()
  {
    std::lock_guard<std::mutex> lockGuard(mutex);

    initialConditionsTimer.reset();
    stepsCounter = 0;
    stepsTimer.reset();
    algebraicVariablesTimer.reset();
    residualsTimer.reset();
    partialDerivativesTimer.reset();
    copyVarsFromMARCOTimer.reset();
    copyVarsIntoMARCOTimer.reset();
  }

  void IDAProfiler::print() const
  {
    std::lock_guard<std::mutex> lockGuard(mutex);

    std::cerr << "Time spent on computing the initial conditions: " << initialConditionsTimer.totalElapsedTime() << " ms\n";
    std::cerr << "Number of IDA steps: " << stepsCounter << "\n";
    std::cerr << "Time spent on IDA steps: " << stepsTimer.totalElapsedTime() << " ms\n";
    std::cerr << "Time spent on computing the algebraic variables: " << algebraicVariablesTimer.totalElapsedTime() << " ms\n";
    std::cerr << "Time spent on computing the residuals: " << residualsTimer.totalElapsedTime() << " ms\n";
    std::cerr << "Time spent on computing the partial derivatives: " << partialDerivativesTimer.totalElapsedTime() << " ms\n";
    std::cerr << "Time spent on copying the variables from MARCO: " << copyVarsFromMARCOTimer.totalElapsedTime() << " ms\n";
    std::cerr << "Time spent on copying the variables into MARCO: " << copyVarsIntoMARCOTimer.totalElapsedTime() << " ms\n";
  }

  void IDAProfiler::incrementStepsCounter()
  {
    std::lock_guard<std::mutex> lockGuard(mutex);
    ++stepsCounter;
  }

  IDAProfiler& idaProfiler()
  {
    static IDAProfiler obj;
    return obj;
  }
}

#endif // MARCO_PROFILING
