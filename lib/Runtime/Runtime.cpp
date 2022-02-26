#include "marco/Runtime/Profiling.h"
#include "marco/Runtime/Runtime.h"
#include "marco/Runtime/IDA.h"

namespace
{
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
