#ifndef MARCO_RUNTIME_SOLVERS_IDA_PROFILER_H
#define MARCO_RUNTIME_SOLVERS_IDA_PROFILER_H

#ifdef MARCO_PROFILING

#include "marco/Runtime/Profiling/Profiler.h"
#include "marco/Runtime/Profiling/Timer.h"
#include <mutex>

namespace marco::runtime::profiling
{
  class IDAProfiler : public Profiler
  {
    public:
      IDAProfiler();

      void reset() override;

      void print() const override;

    public:
      Timer initialConditionsTimer;
      Timer stepsTimer;
      Timer algebraicVariablesTimer;

      mutable std::mutex mutex;
  };

  IDAProfiler& idaProfiler();
}

#define IDA_PROFILER_IC_START ::marco::runtime::profiling::idaProfiler().initialConditionsTimer.start()
#define IDA_PROFILER_IC_STOP ::marco::runtime::profiling::idaProfiler().initialConditionsTimer.stop()

#define IDA_PROFILER_STEP_START ::marco::runtime::profiling::idaProfiler().stepsTimer.start()
#define IDA_PROFILER_STEP_STOP ::marco::runtime::profiling::idaProfiler().stepsTimer.stop()

#define IDA_PROFILER_ALGEBRAIC_VARS_START ::marco::runtime::profiling::idaProfiler().algebraicVariablesTimer.start()
#define IDA_PROFILER_ALGEBRAIC_VARS_STOP ::marco::runtime::profiling::idaProfiler().algebraicVariablesTimer.stop()

#else

#define IDA_PROFILER_DO_NOTHING static_assert(true)

#define IDA_PROFILER_IC_START IDA_PROFILER_DO_NOTHING
#define IDA_PROFILER_IC_STOP IDA_PROFILER_DO_NOTHING

#define IDA_PROFILER_STEP_START IDA_PROFILER_DO_NOTHING
#define IDA_PROFILER_STEP_STOP IDA_PROFILER_DO_NOTHING

#define IDA_PROFILER_ALGEBRAIC_VARS_START IDA_PROFILER_DO_NOTHING
#define IDA_PROFILER_ALGEBRAIC_VARS_STOP IDA_PROFILER_DO_NOTHING

#endif // MARCO_PROFILING

#endif // MARCO_RUNTIME_SOLVERS_IDA_PROFILER_H
