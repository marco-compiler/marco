#ifndef MARCO_RUNTIME_SIMULATION_PROFILER_H
#define MARCO_RUNTIME_SIMULATION_PROFILER_H

#ifdef MARCO_PROFILING

#include "marco/Runtime/Profiling/Profiling.h"
#include "marco/Runtime/Profiling/Timer.h"
#include <mutex>

namespace marco::runtime::profiling
{
  class SimulationProfiler : public Profiler
  {
    public:
      SimulationProfiler();

      void reset() override;

      void print() const override;

    public:
      Timer commandLineArgs;
      Timer initialization;
      Timer printing;

      mutable std::mutex mutex;
  };

  SimulationProfiler& simulationProfiler();
}

#define SIMULATION_PROFILER_ARG_START ::marco::runtime::profiling::simulationProfiler().commandLineArgs.start()
#define SIMULATION_PROFILER_ARG_STOP ::marco::runtime::profiling::simulationProfiler().commandLineArgs.stop()

#define SIMULATION_PROFILER_INIT_START ::marco::runtime::profiling::simulationProfiler().initialization.start()
#define SIMULATION_PROFILER_INIT_STOP ::marco::runtime::profiling::simulationProfiler().initialization.stop()

#define SIMULATION_PROFILER_PRINTING_START ::marco::runtime::profiling::simulationProfiler().printing.start()
#define SIMULATION_PROFILER_PRINTING_STOP ::marco::runtime::profiling::simulationProfiler().printing.stop()

#else

#define SIMULATION_PROFILER_DO_NOTHING static_assert(true)

#define SIMULATION_PROFILER_ARG_START SIMULATION_PROFILER_DO_NOTHING
#define SIMULATION_PROFILER_ARG_STOP SIMULATION_PROFILER_DO_NOTHING

#define SIMULATION_PROFILER_INIT_START SIMULATION_PROFILER_DO_NOTHING
#define SIMULATION_PROFILER_INIT_STOP SIMULATION_PROFILER_DO_NOTHING

#define SIMULATION_PROFILER_PRINTING_START SIMULATION_PROFILER_DO_NOTHING
#define SIMULATION_PROFILER_PRINTING_STOP SIMULATION_PROFILER_DO_NOTHING

#endif // MARCO_PROFILING

#endif // MARCO_RUNTIME_SIMULATION_PROFILER_H
