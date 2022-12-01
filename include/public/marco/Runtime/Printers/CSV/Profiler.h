#ifndef MARCO_RUNTIME_PRINTING_PRINTPROFILER_H
#define MARCO_RUNTIME_PRINTING_PRINTPROFILER_H

#ifdef MARCO_PROFILING

#include "marco/Runtime/Profiling/Profiling.h"
#include "marco/Runtime/Profiling/Timer.h"
#include <mutex>

namespace marco::runtime::profiling
{
  class PrintProfiler : public Profiler
  {
    public:
      PrintProfiler();

      void reset() override;

      void print() const override;

    public:
      Timer booleanValues;
      Timer integerValues;
      Timer floatValues;
      Timer stringValues;

      mutable std::mutex mutex;
  };

  PrintProfiler& printProfiler();
}

#define PRINT_PROFILER_BOOL_START ::marco::runtime::profiling::printProfiler().booleanValues.start()
#define PRINT_PROFILER_BOOL_STOP ::marco::runtime::profiling::printProfiler().booleanValues.stop()

#define PRINT_PROFILER_INT_START ::marco::runtime::profiling::printProfiler().integerValues.start()
#define PRINT_PROFILER_INT_STOP ::marco::runtime::profiling::printProfiler().integerValues.stop()

#define PRINT_PROFILER_FLOAT_START ::marco::runtime::profiling::printProfiler().floatValues.start()
#define PRINT_PROFILER_FLOAT_STOP ::marco::runtime::profiling::printProfiler().floatValues.stop()

#define PRINT_PROFILER_STRING_START ::marco::runtime::profiling::printProfiler().stringValues.start()
#define PRINT_PROFILER_STRING_STOP ::marco::runtime::profiling::printProfiler().stringValues.start()

#else

#define PRINT_PROFILER_DO_NOTHING static_assert(true)

#define PRINT_PROFILER_BOOL_START PRINT_PROFILER_DO_NOTHING
#define PRINT_PROFILER_BOOL_STOP PRINT_PROFILER_DO_NOTHING

#define PRINT_PROFILER_INT_START PRINT_PROFILER_DO_NOTHING
#define PRINT_PROFILER_INT_STOP PRINT_PROFILER_DO_NOTHING

#define PRINT_PROFILER_FLOAT_START PRINT_PROFILER_DO_NOTHING
#define PRINT_PROFILER_FLOAT_STOP PRINT_PROFILER_DO_NOTHING

#define PRINT_PROFILER_STRING_START PRINT_PROFILER_DO_NOTHING
#define PRINT_PROFILER_STRING_STOP PRINT_PROFILER_DO_NOTHING

#endif // MARCO_PROFILING

#endif // MARCO_RUNTIME_PRINTING_PRINTPROFILER_H
