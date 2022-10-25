#include "marco/Runtime/Printers/CSV/Profiler.h"
#include <iostream>

#ifdef MARCO_PROFILING

namespace marco::runtime::profiling
{
  PrintProfiler::PrintProfiler()
      : Profiler("Simulation data printing")
  {
    registerProfiler(*this);
  }

  void PrintProfiler::reset()
  {
    booleanValues.reset();
    integerValues.reset();
    floatValues.reset();
    stringValues.reset();
  }

  void PrintProfiler::print() const
  {
    std::cerr << "Time spent on printing boolean values: " << booleanValues.totalElapsedTime() << " ms\n";
    std::cerr << "Time spent on printing integer values: " << integerValues.totalElapsedTime() << " ms\n";
    std::cerr << "Time spent on printing float values: " << floatValues.totalElapsedTime() << " ms\n";
    std::cerr << "Time spent on printing strings: " << stringValues.totalElapsedTime() << " ms\n";
  }

  PrintProfiler& printProfiler()
  {
    static PrintProfiler obj;
    return obj;
  }
}

#endif
