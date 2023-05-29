#include "marco/Runtime/Support/Profiler.h"
#include "marco/Runtime/Profiling/Profiling.h"
#include <iostream>

#ifdef MARCO_PROFILING

namespace marco::runtime::profiling
{
  SupportProfiler::SupportProfiler()
      : Profiler("Support")
  {
    registerProfiler(*this);
  }

  void SupportProfiler::reset()
  {
    std::lock_guard<std::mutex> lockGuard(mutex);

    sinFunction.reset();
    stdSinCounter = 0;

    cosFunction.reset();
    stdCosCounter = 0;
  }

  void SupportProfiler::print() const
  {
    std::lock_guard<std::mutex> lockGuard(mutex);

    std::cerr << "Number of 'std::sin' function calls: " << stdSinCounter << "\n";
    std::cerr << "Time spent on computing the 'sin' function: " << sinFunction.totalElapsedTime() << " ms\n";
    std::cerr << "Number of 'std::cos' function calls: " << stdCosCounter << "\n";
    std::cerr << "Time spent on computing the 'cos' function: " << cosFunction.totalElapsedTime() << " ms\n";
  }

  void SupportProfiler::incrementStdSinCounter()
  {
    std::lock_guard<std::mutex> lockGuard(mutex);
    ++stdSinCounter;
  }

  void SupportProfiler::incrementStdCosCounter()
  {
    std::lock_guard<std::mutex> lockGuard(mutex);
    ++stdCosCounter;
  }

  SupportProfiler& supportProfiler()
  {
    static SupportProfiler obj;
    return obj;
  }
}

#endif // MARCO_PROFILING
