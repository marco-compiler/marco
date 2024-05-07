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
    residualsCallCounter = 0;
    residualsTimer.reset();
    partialDerivativesCallCounter = 0;
    partialDerivativesTimer.reset();
    copyVarsFromMARCOTimer.reset();
    copyVarsIntoMARCOTimer.reset();
  }

  void IDAProfiler::print() const
  {
    std::lock_guard<std::mutex> lockGuard(mutex);

    std::cerr << "Time spent on computing the initial conditions: "
              << initialConditionsTimer.totalElapsedTime<std::milli>() << " ms"
              << std::endl;

    std::cerr << "Number of IDA steps: " << stepsCounter << std::endl;

    std::cerr << "Time spent on IDA steps: "
              << stepsTimer.totalElapsedTime<std::milli>() << " ms"
              << std::endl;

    std::cerr << "Time spent on computing the algebraic variables: "
              << algebraicVariablesTimer.totalElapsedTime<std::milli>()
              << " ms" << std::endl;

    std::cerr << "Number of computations of the residuals: "
              << residualsCallCounter << std::endl;

    std::cerr << "Time spent on computing the residuals: "
              << residualsTimer.totalElapsedTime<std::milli>() << " ms"
              << std::endl;

    std::cerr << "Number of computations of the partial derivatives: "
              << partialDerivativesCallCounter << std::endl;

    std::cerr << "Time spent on computing the partial derivatives: "
              << partialDerivativesTimer.totalElapsedTime<std::milli>()
              << " ms" << std::endl;

    std::cerr << "Time spent on copying the variables from MARCO: "
              << copyVarsFromMARCOTimer.totalElapsedTime<std::milli>()
              << " ms" << std::endl;

    std::cerr << "Time spent on copying the variables into MARCO: "
              << copyVarsIntoMARCOTimer.totalElapsedTime<std::milli>()
              << " ms" << std::endl;
  }

  void IDAProfiler::incrementStepsCounter()
  {
    std::lock_guard<std::mutex> lockGuard(mutex);
    ++stepsCounter;
  }

  void IDAProfiler::incrementResidualsCallCounter()
  {
    std::lock_guard<std::mutex> lockGuard(mutex);
    ++residualsCallCounter;
  }

  void IDAProfiler::incrementPartialDerivativesCallCounter()
  {
    std::lock_guard<std::mutex> lockGuard(mutex);
    ++partialDerivativesCallCounter;
  }

  IDAProfiler& idaProfiler()
  {
    static IDAProfiler obj;
    return obj;
  }
}

#endif // MARCO_PROFILING
