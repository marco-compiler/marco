#include "marco/Runtime/Profiling.h"
#include "marco/Runtime/Runtime.h"

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
extern "C" void updateNonStateVariables(void* data);
extern "C" void updateStateVariables(void* data);
extern "C" bool incrementTime(void* data);
extern "C" void deinit(void* data);

extern "C" void printHeader(void* data);
extern "C" void print(void* data);

void runSimulation()
{
  runtimeInit();
  void* data = init();

  // TODO check IDA initialization through static checker
  // Same for IDA step

  printHeader(data);

  updateNonStateVariables(data);
  bool continueSimulation;

  do {
    print(data);
    updateStateVariables(data);
    continueSimulation = incrementTime(data);
    updateNonStateVariables(data);
  } while (continueSimulation);

  deinit(data);
  runtimeDeinit();
}
