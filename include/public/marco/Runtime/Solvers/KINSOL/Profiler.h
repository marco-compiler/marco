#ifndef MARCO_RUNTIME_SOLVERS_KINSOL_PROFILER_H
#define MARCO_RUNTIME_SOLVERS_KINSOL_PROFILER_H

#ifdef MARCO_PROFILING

#include "marco/Runtime/Profiling/Profiler.h"
#include "marco/Runtime/Profiling/Timer.h"

namespace marco::runtime::profiling
{
  class KINSOLProfiler : public Profiler
  {
    public:
        KINSOLProfiler();

      void reset() override;

      void print() const override;

    public:
      Timer stepsTimer;
  };

  KINSOLProfiler& kinsolProfiler();
}

#define KINSOL_PROFILER_STEP_START ::marco::runtime::profiling::kinsolProfiler().stepsTimer.start()
#define KINSOL_PROFILER_STEP_STOP ::marco::runtime::profiling::kinsolProfiler().stepsTimer.stop()

#else

#define KINSOL_PROFILER_DO_NOTHING static_assert(true)

#define KINSOL_PROFILER_STEP_START KINSOL_PROFILER_DO_NOTHING
#define KINSOL_PROFILER_STEP_STOP KINSOL_PROFILER_DO_NOTHING

#endif // MARCO_PROFILING

#endif // MARCO_RUNTIME_SOLVERS_KINSOL_PROFILER_H
