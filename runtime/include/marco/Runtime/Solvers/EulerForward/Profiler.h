#ifndef MARCO_RUNTIME_SOLVERS_EULERFORWARD_PROFILER_H
#define MARCO_RUNTIME_SOLVERS_EULERFORWARD_PROFILER_H

#ifdef MARCO_PROFILING

#include "marco/Runtime/Profiling/Profiling.h"
#include "marco/Runtime/Profiling/Timer.h"
#include <mutex>

namespace marco::runtime::profiling
{
  class EulerForwardProfiler : public Profiler
  {
    public:
      EulerForwardProfiler();

      void reset() override;

      void print() const override;

    public:
      Timer initialConditions;
      Timer stateVariables;
      Timer nonStateVariables;

      mutable std::mutex mutex;
  };

  EulerForwardProfiler& eulerForwardProfiler();
}

#define EULER_FORWARD_PROFILER_IC_START ::marco::runtime::profiling::eulerForwardProfiler().initialConditions.start();
#define EULER_FORWARD_PROFILER_IC_STOP ::marco::runtime::profiling::eulerForwardProfiler().initialConditions.stop();

#define EULER_FORWARD_PROFILER_STATEVAR_START ::marco::runtime::profiling::eulerForwardProfiler().stateVariables.start();
#define EULER_FORWARD_PROFILER_STATEVAR_STOP ::marco::runtime::profiling::eulerForwardProfiler().stateVariables.stop();

#define EULER_FORWARD_PROFILER_NONSTATEVAR_START ::marco::runtime::profiling::eulerForwardProfiler().nonStateVariables.start();
#define EULER_FORWARD_PROFILER_NONSTATEVAR_STOP ::marco::runtime::profiling::eulerForwardProfiler().nonStateVariables.stop();

#else

#define EULER_FORWARD_PROFILER_IC_START static_assert(true)
#define EULER_FORWARD_PROFILER_IC_STOP static_assert(true)

#define EULER_FORWARD_PROFILER_STATEVAR_START static_assert(true)
#define EULER_FORWARD_PROFILER_STATEVAR_STOP static_assert(true)

#define EULER_FORWARD_PROFILER_NONSTATEVAR_START static_assert(true)
#define EULER_FORWARD_PROFILER_NONSTATEVAR_STOP static_assert(true)

#endif // MARCO_PROFILING

#endif // MARCO_RUNTIME_SOLVERS_EULERFORWARD_PROFILER_H
