#include "marco/Runtime/Runtime.h"
#include "marco/Runtime/IO.h"
#include "marco/Runtime/Profiling.h"
#include "argh.h"

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

namespace
{
  void printHelp()
  {
    std::cout << "Modelica simulation.\n";
    std::cout << "Generated with MARCO compiler.\n\n";

    std::cout << "OPTIONS:\n";
    std::cout << "  --help                     Display the available options.\n";
    std::cout << "  --scientific-notation      Print the values using the scientific notation.\n";
    std::cout << "  --precision=<value>        Set the number of decimals to be printed.\n";
  }
}

[[maybe_unused]] int runSimulation(int argc, char* argv[])
{
  // Parse the command-line arguments
  argh::parser cmdl(argc, argv);

  ioConfig().scientificNotation = cmdl["scientific-notation"];
  cmdl("precision", ioConfig().precision) >> ioConfig().precision;

  if (cmdl["help"]) {
    printHelp();
    return 0;
  }

  // Initialize the runtime library
  runtimeInit();

  // Run the simulation
  void* data = init();

  // TODO check IDA initialization through static checker
  // Same for IDA step

  printHeader(data);

  updateNonStateVariables(data);
  print(data);

  bool continueSimulation;

  do {
    updateStateVariables(data);
    continueSimulation = incrementTime(data);
    updateNonStateVariables(data);
    print(data);
  } while (continueSimulation);

  deinit(data);

  // De-initialize the runtime library
  runtimeDeinit();

  return 0;
}
