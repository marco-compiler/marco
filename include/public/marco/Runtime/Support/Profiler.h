#ifndef MARCO_RUNTIME_SUPPORT_PROFILER_H
#define MARCO_RUNTIME_SUPPORT_PROFILER_H

#ifdef MARCO_PROFILING

#include "marco/Runtime/Profiling/Profiling.h"
#include "marco/Runtime/Profiling/Timer.h"
#include <mutex>

namespace marco::runtime::profiling
{
  class SupportProfiler : public Profiler
  {
    public:
      SupportProfiler();

      void reset() override;

      void print() const override;

      void incrementSinCounter();

      void incrementCosCounter();

    public:
      size_t sinCounter;
      Timer sinFunction;

      size_t cosCounter;
      Timer cosFunction;

      mutable std::mutex mutex;
  };

  SupportProfiler& supportProfiler();
}

#define SUPPORT_PROFILER_SINCOUNTER_INCREMENT ::marco::runtime::profiling::supportProfiler().incrementSinCounter()

#define SUPPORT_PROFILER_SINFUNCTION_START ::marco::runtime::profiling::supportProfiler().sinFunction.start()
#define SUPPORT_PROFILER_SINFUNCTION_STOP ::marco::runtime::profiling::supportProfiler().sinFunction.stop()

#define SUPPORT_PROFILER_COSCOUNTER_INCREMENT ::marco::runtime::profiling::supportProfiler().incrementCosCounter()

#define SUPPORT_PROFILER_COSFUNCTION_START ::marco::runtime::profiling::supportProfiler().cosFunction.start()
#define SUPPORT_PROFILER_COSFUNCTION_STOP ::marco::runtime::profiling::supportProfiler().cosFunction.stop()

#else

#define SUPPORT_PROFILER_DO_NOTHING static_assert(true)

#define SUPPORT_PROFILER_SINCOUNTER_INCREMENT SUPPORT_PROFILER_DO_NOTHING

#define SUPPORT_PROFILER_SINFUNCTION_START SUPPORT_PROFILER_DO_NOTHING
#define SUPPORT_PROFILER_SINFUNCTION_STOP SUPPORT_PROFILER_DO_NOTHING

#define SUPPORT_PROFILER_COSCOUNTER_INCREMENT SUPPORT_PROFILER_DO_NOTHING

#define SUPPORT_PROFILER_COSFUNCTION_START SUPPORT_PROFILER_DO_NOTHING
#define SUPPORT_PROFILER_COSFUNCTION_STOP SUPPORT_PROFILER_DO_NOTHING

#endif // MARCO_PROFILING

#endif // MARCO_RUNTIME_SUPPORT_PROFILER_H
