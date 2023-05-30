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

      void incrementStepsCounter();

    public:
      Timer initialConditionsTimer;
      int64_t stepsCounter;
      Timer stepsTimer;
      Timer algebraicVariablesTimer;
      Timer residualsTimer;
      Timer partialDerivativesTimer;
      Timer copyVarsFromMARCOTimer;
      Timer copyVarsIntoMARCOTimer;

      mutable std::mutex mutex;
  };

  IDAProfiler& idaProfiler();
}

#define IDA_PROFILER_IC_START ::marco::runtime::profiling::idaProfiler().initialConditionsTimer.start()
#define IDA_PROFILER_IC_STOP ::marco::runtime::profiling::idaProfiler().initialConditionsTimer.stop()

#define IDA_PROFILER_STEPS_COUNTER_INCREMENT ::marco::runtime::profiling::idaProfiler().incrementStepsCounter()

#define IDA_PROFILER_STEP_START ::marco::runtime::profiling::idaProfiler().stepsTimer.start()
#define IDA_PROFILER_STEP_STOP ::marco::runtime::profiling::idaProfiler().stepsTimer.stop()

#define IDA_PROFILER_ALGEBRAIC_VARS_START ::marco::runtime::profiling::idaProfiler().algebraicVariablesTimer.start()
#define IDA_PROFILER_ALGEBRAIC_VARS_STOP ::marco::runtime::profiling::idaProfiler().algebraicVariablesTimer.stop()

#define IDA_PROFILER_RESIDUALS_START ::marco::runtime::profiling::idaProfiler().residualsTimer.start()
#define IDA_PROFILER_RESIDUALS_STOP ::marco::runtime::profiling::idaProfiler().residualsTimer.stop()

#define IDA_PROFILER_PARTIAL_DERIVATIVES_START ::marco::runtime::profiling::idaProfiler().partialDerivativesTimer.start()
#define IDA_PROFILER_PARTIAL_DERIVATIVES_STOP ::marco::runtime::profiling::idaProfiler().partialDerivativesTimer.stop()

#define IDA_PROFILER_COPY_VARS_FROM_MARCO_START ::marco::runtime::profiling::idaProfiler().copyVarsFromMARCOTimer.start()
#define IDA_PROFILER_COPY_VARS_FROM_MARCO_STOP ::marco::runtime::profiling::idaProfiler().copyVarsFromMARCOTimer.stop()

#define IDA_PROFILER_COPY_VARS_INTO_MARCO_START ::marco::runtime::profiling::idaProfiler().copyVarsIntoMARCOTimer.start()
#define IDA_PROFILER_COPY_VARS_INTO_MARCO_STOP ::marco::runtime::profiling::idaProfiler().copyVarsIntoMARCOTimer.stop()

#else

#define IDA_PROFILER_DO_NOTHING static_assert(true)

#define IDA_PROFILER_IC_START IDA_PROFILER_DO_NOTHING
#define IDA_PROFILER_IC_STOP IDA_PROFILER_DO_NOTHING

IDA_PROFILER_STEPS_COUNTER_INCREMENT IDA_PROFILER_DO_NOTHING

#define IDA_PROFILER_STEP_START IDA_PROFILER_DO_NOTHING
#define IDA_PROFILER_STEP_STOP IDA_PROFILER_DO_NOTHING

#define IDA_PROFILER_ALGEBRAIC_VARS_START IDA_PROFILER_DO_NOTHING
#define IDA_PROFILER_ALGEBRAIC_VARS_STOP IDA_PROFILER_DO_NOTHING

#define IDA_PROFILER_RESIDUALS_START IDA_PROFILER_DO_NOTHING
#define IDA_PROFILER_RESIDUALS_STOP IDA_PROFILER_DO_NOTHING

#define IDA_PROFILER_PARTIAL_DERIVATIVES_START IDA_PROFILER_DO_NOTHING
#define IDA_PROFILER_PARTIAL_DERIVATIVES_STOP IDA_PROFILER_DO_NOTHING

#define IDA_PROFILER_COPY_VARS_FROM_MARCO_START IDA_PROFILER_DO_NOTHING
#define IDA_PROFILER_COPY_VARS_FROM_MARCO_STOP IDA_PROFILER_DO_NOTHING

#define IDA_PROFILER_COPY_VARS_INTO_MARCO_START IDA_PROFILER_DO_NOTHING
#define IDA_PROFILER_COPY_VARS_INTO_MARCO_STOP IDA_PROFILER_DO_NOTHING

#endif // MARCO_PROFILING

#endif // MARCO_RUNTIME_SOLVERS_IDA_PROFILER_H
