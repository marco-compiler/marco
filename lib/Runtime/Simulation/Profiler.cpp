#include "marco/Runtime/Simulation/Profiler.h"
#include <iostream>

#ifdef MARCO_PROFILING

namespace marco::runtime::profiling
{
  SimulationProfiler::SimulationProfiler()
      : Profiler("Simulation")
  {
    registerProfiler(*this);
  }

  void SimulationProfiler::reset()
  {
    commandLineArgs.reset();
    initialization.reset();
    printing.reset();
  }

  void SimulationProfiler::print() const
  {
    std::cerr << "Time spent on command-line arguments processing: " << commandLineArgs.totalElapsedTime() << " ms\n";
    std::cerr << "Time spent on initialization: " << initialization.totalElapsedTime() << " ms\n";
    std::cerr << "Time spent on values printing: " << printing.totalElapsedTime() << " ms\n";
  }

  SimulationProfiler& simulationProfiler()
  {
    static SimulationProfiler obj;
    return obj;
  }
}

#endif
