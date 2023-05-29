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
    sinCounter = 0;

    cosFunction.reset();
    cosCounter = 0;
  }

  void SupportProfiler::print() const
  {
    std::lock_guard<std::mutex> lockGuard(mutex);

    std::cerr << "Number of 'sin' function calls: " << sinCounter << "\n";
    std::cerr << "Time spent on computing the 'sin' function: " << sinFunction.totalElapsedTime() << " ms\n";
    std::cerr << "Number of 'cos' function calls: " << cosCounter << "\n";
    std::cerr << "Time spent on computing the 'cos' function: " << cosFunction.totalElapsedTime() << " ms\n";
  }

  void SupportProfiler::incrementSinCounter()
  {
    std::lock_guard<std::mutex> lockGuard(mutex);
    ++sinCounter;
  }

  void SupportProfiler::incrementCosCounter()
  {
    std::lock_guard<std::mutex> lockGuard(mutex);
    ++cosCounter;
  }

  SupportProfiler& supportProfiler()
  {
    static SupportProfiler obj;
    return obj;
  }
}

#endif // MARCO_PROFILING
