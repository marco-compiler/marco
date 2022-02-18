#include "marco/runtime/IDA.h"
#include "marco/runtime/Profiling.h"
#include "marco/runtime/Runtime.h"

static void runtimeInit()
{
#ifdef MARCO_PROFILING
  profilingInit();
#endif
}

static void runtimeDeinit()
{
#ifdef MARCO_PROFILING
  printProfilingStats();
#endif
}

extern "C" void* init();
extern "C" bool step(void* data);
extern "C" void updateStateVariables(void* data);
extern "C" void deinit(void* data);

extern "C" void printHeader(void* data);
extern "C" void print(void* data);

void runSimulation() {
  runtimeInit();
  void* data = init();

  printHeader(data);
  print(data);

  while (step(data)) {
    print(data);
    updateStateVariables(data);
  }

  deinit(data);
  runtimeDeinit();
}
