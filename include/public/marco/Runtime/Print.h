#ifndef MARCO_RUNTIME_PRINT_H
#define MARCO_RUNTIME_PRINT_H

#include "marco/Runtime/Mangling.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cstdint>
#include <iostream>

struct PrinterConfig
{
  bool scientificNotation = false;
  unsigned int precision = 9;
};

PrinterConfig& printerConfig();

//===----------------------------------------------------------------------===//
// CLI
//===----------------------------------------------------------------------===//

#ifdef MARCO_CLI

#include "marco/Runtime/CLI.h"

namespace marco::runtime::formatting
{
  std::unique_ptr<cli::Category> getCLIOptionsCategory();
}

#endif

//===----------------------------------------------------------------------===//
// Profiling
//===----------------------------------------------------------------------===//

#ifdef MARCO_PROFILING
#include "marco/Runtime/Profiling.h"

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
};

PrintProfiler& printProfiler();

#define PRINT_PROFILER_BOOL_START ::printProfiler().booleanValues.start()
#define PRINT_PROFILER_BOOL_STOP ::printProfiler().booleanValues.stop()

#define PRINT_PROFILER_INT_START ::printProfiler().integerValues.start()
#define PRINT_PROFILER_INT_STOP ::printProfiler().integerValues.stop()

#define PRINT_PROFILER_FLOAT_START ::printProfiler().floatValues.start()
#define PRINT_PROFILER_FLOAT_STOP ::printProfiler().floatValues.stop()

#define PRINT_PROFILER_STRING_START ::printProfiler().stringValues.start()
#define PRINT_PROFILER_STRING_STOP ::printProfiler().stringValues.start()

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

#endif

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(print, void, bool)
RUNTIME_FUNC_DECL(print, void, int32_t)
RUNTIME_FUNC_DECL(print, void, int64_t)
RUNTIME_FUNC_DECL(print, void, float)
RUNTIME_FUNC_DECL(print, void, double)

RUNTIME_FUNC_DECL(print, void, ARRAY(bool))
RUNTIME_FUNC_DECL(print, void, ARRAY(int32_t))
RUNTIME_FUNC_DECL(print, void, ARRAY(int64_t))
RUNTIME_FUNC_DECL(print, void, ARRAY(float))
RUNTIME_FUNC_DECL(print, void, ARRAY(double))

#endif // MARCO_RUNTIME_PRINT_H
