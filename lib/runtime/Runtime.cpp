#include <marco/runtime/Profiling.h>
#include <marco/runtime/Runtime.h>

void runtimeInit()
{
#ifdef MARCO_PROFILING
  profilingInit();
#endif
}

void runtimeDeinit()
{
#ifdef MARCO_PROFILING
  printProfilingStats();
#endif
}
