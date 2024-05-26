#ifndef MARCO_RUNTIME_SOLVERS_KINSOL_PROFILER_H
#define MARCO_RUNTIME_SOLVERS_KINSOL_PROFILER_H

#ifdef MARCO_PROFILING

#include "marco/Runtime/Profiling/Profiler.h"
#include "marco/Runtime/Profiling/Timer.h"
#include <mutex>

namespace marco::runtime::profiling
{
  class KINSOLProfiler : public Profiler
  {
    public:
      KINSOLProfiler();

      void reset() override;

      void print() const override;

      void incrementResidualsCallCounter();

      void incrementPartialDerivativesCallCounter();

    public:
      int64_t residualsCallCounter;
      Timer residualsTimer;
      int64_t partialDerivativesCallCounter;
      Timer partialDerivativesTimer;
      Timer copyVarsFromMARCOTimer;
      Timer copyVarsIntoMARCOTimer;

      mutable std::mutex mutex;
  };

  KINSOLProfiler& kinsolProfiler();
}

#define KINSOL_PROFILER_IC_START ::marco::runtime::profiling::kinsolProfiler().initialConditionsTimer.start()
#define KINSOL_PROFILER_IC_STOP ::marco::runtime::profiling::kinsolProfiler().initialConditionsTimer.stop()

#define KINSOL_PROFILER_RESIDUALS_CALL_COUNTER_INCREMENT ::marco::runtime::profiling::kinsolProfiler().incrementResidualsCallCounter()

#define KINSOL_PROFILER_RESIDUALS_START ::marco::runtime::profiling::kinsolProfiler().residualsTimer.start()
#define KINSOL_PROFILER_RESIDUALS_STOP ::marco::runtime::profiling::kinsolProfiler().residualsTimer.stop()

#define KINSOL_PROFILER_PARTIAL_DERIVATIVES_CALL_COUNTER_INCREMENT ::marco::runtime::profiling::kinsolProfiler().incrementPartialDerivativesCallCounter()

#define KINSOL_PROFILER_PARTIAL_DERIVATIVES_START ::marco::runtime::profiling::kinsolProfiler().partialDerivativesTimer.start()
#define KINSOL_PROFILER_PARTIAL_DERIVATIVES_STOP ::marco::runtime::profiling::kinsolProfiler().partialDerivativesTimer.stop()

#define KINSOL_PROFILER_COPY_VARS_FROM_MARCO_START ::marco::runtime::profiling::kinsolProfiler().copyVarsFromMARCOTimer.start()
#define KINSOL_PROFILER_COPY_VARS_FROM_MARCO_STOP ::marco::runtime::profiling::kinsolProfiler().copyVarsFromMARCOTimer.stop()

#define KINSOL_PROFILER_COPY_VARS_INTO_MARCO_START ::marco::runtime::profiling::kinsolProfiler().copyVarsIntoMARCOTimer.start()
#define KINSOL_PROFILER_COPY_VARS_INTO_MARCO_STOP ::marco::runtime::profiling::kinsolProfiler().copyVarsIntoMARCOTimer.stop()


#else

#define KINSOL_PROFILER_DO_NOTHING static_assert(true)

#define KINSOL_PROFILER_IC_START KINSOL_PROFILER_DO_NOTHING
#define KINSOL_PROFILER_IC_STOP KINSOL_PROFILER_DO_NOTHING

#define KINSOL_PROFILER_RESIDUALS_CALL_COUNTER_INCREMENT KINSOL_PROFILER_DO_NOTHING

#define KINSOL_PROFILER_RESIDUALS_START KINSOL_PROFILER_DO_NOTHING
#define KINSOL_PROFILER_RESIDUALS_STOP KINSOL_PROFILER_DO_NOTHING

#define KINSOL_PROFILER_PARTIAL_DERIVATIVES_CALL_COUNTER_INCREMENT KINSOL_PROFILER_DO_NOTHING

#define KINSOL_PROFILER_PARTIAL_DERIVATIVES_START KINSOL_PROFILER_DO_NOTHING
#define KINSOL_PROFILER_PARTIAL_DERIVATIVES_STOP KINSOL_PROFILER_DO_NOTHING

#define KINSOL_PROFILER_COPY_VARS_FROM_MARCO_START KINSOL_PROFILER_DO_NOTHING
#define KINSOL_PROFILER_COPY_VARS_FROM_MARCO_STOP KINSOL_PROFILER_DO_NOTHING

#define KINSOL_PROFILER_COPY_VARS_INTO_MARCO_START KINSOL_PROFILER_DO_NOTHING
#define KINSOL_PROFILER_COPY_VARS_INTO_MARCO_STOP KINSOL_PROFILER_DO_NOTHING

#endif // MARCO_PROFILING

#endif // MARCO_RUNTIME_SOLVERS_KINSOL_PROFILER_H
