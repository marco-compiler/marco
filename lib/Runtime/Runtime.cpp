#include "marco/Runtime/Runtime.h"
#include "marco/Runtime/Print.h"
#include "argh.h"

// Functions defined by the compiled model module
extern "C" void* init();
extern "C" void updateNonStateVariables(void* data);
extern "C" void updateStateVariables(void* data);
extern "C" bool incrementTime(void* data);
extern "C" void deinit(void* data);

extern "C" void printHeader(void* data);
extern "C" void print(void* data);

#ifdef MARCO_PROFILING

#include "marco/Runtime/Profiling.h"
#include <iostream>

namespace
{
  class SimulationProfiler : public Profiler
  {
    public:
      SimulationProfiler() : Profiler("Simulation")
      {
        registerProfiler(*this);
      }

      void reset() override
      {
        commandLineArgs.reset();
        initialization.reset();
        nonStateVariables.reset();
        stateVariables.reset();
        printing.reset();
      }

      void print() const override
      {
        std::cerr << "Time spent on command-line arguments processing: " << commandLineArgs.totalElapsedTime() << " ms\n";
        std::cerr << "Time spent on initialization: " << initialization.totalElapsedTime() << " ms\n";
        std::cerr << "Time spent on non-state variables computation: " << nonStateVariables.totalElapsedTime() << " ms\n";
        std::cerr << "Time spent on state variables computation: " << stateVariables.totalElapsedTime() << " ms\n";
        std::cerr << "Time spent on values printing: " << stateVariables.totalElapsedTime() << " ms\n";
      }

    public:
      Timer commandLineArgs;
      Timer initialization;
      Timer nonStateVariables;
      Timer stateVariables;
      Timer printing;
  };

  SimulationProfiler& profiler()
  {
    static SimulationProfiler obj;
    return obj;
  }
}

  #define PROFILER_ARG_START ::profiler().commandLineArgs.start()
  #define PROFILER_ARG_STOP ::profiler().commandLineArgs.stop()

  #define PROFILER_INIT_START ::profiler().initialization.start()
  #define PROFILER_INIT_STOP ::profiler().initialization.stop()

  #define PROFILER_NONSTATEVAR_START ::profiler().nonStateVariables.start()
  #define PROFILER_NONSTATEVAR_STOP ::profiler().nonStateVariables.stop()

  #define PROFILER_STATEVAR_START ::profiler().stateVariables.start()
  #define PROFILER_STATEVAR_STOP ::profiler().stateVariables.start()

  #define PROFILER_PRINTING_START ::profiler().printing.start()
  #define PROFILER_PRINTING_STOP ::profiler().printing.start()

#else
  #define PROFILER_ARG_START
  #define PROFILER_ARG_STOP

  #define PROFILER_INIT_START
  #define PROFILER_INIT_STOP

  #define PROFILER_NONSTATEVAR_START
  #define PROFILER_NONSTATEVAR_STOP

  #define PROFILER_STATEVAR_START
  #define PROFILER_STATEVAR_STOP

  #define PROFILER_PRINTING_START
  #define PROFILER_PRINTING_STOP
#endif

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
  PROFILER_ARG_START;
  argh::parser cmdl(argc, argv);

  printerConfig().scientificNotation = cmdl["scientific-notation"];
  cmdl("precision", printerConfig().precision) >> printerConfig().precision;

  if (cmdl["help"]) {
    printHelp();
    return 0;
  }

  PROFILER_ARG_STOP;

  // Initialize the runtime library
  runtimeInit();

  // Compute the initial values
  PROFILER_INIT_START;
  void* data = init();
  PROFILER_INIT_STOP;

  // TODO check IDA initialization through static checker
  // Same for IDA step

  PROFILER_PRINTING_START;
  printHeader(data);
  PROFILER_PRINTING_STOP;

  // Update the non-state variables.
  PROFILER_NONSTATEVAR_START;
  updateNonStateVariables(data);
  PROFILER_NONSTATEVAR_STOP;

  // Print the data header and the initial values
  PROFILER_PRINTING_START;
  print(data);
  PROFILER_PRINTING_STOP;

  bool continueSimulation;

  do {
    // Compute the next state variables
    PROFILER_STATEVAR_START;
    updateStateVariables(data);
    PROFILER_STATEVAR_STOP;

    // Move to the next step
    PROFILER_NONSTATEVAR_START;
    continueSimulation = incrementTime(data);
    updateNonStateVariables(data);
    PROFILER_NONSTATEVAR_STOP;

    // Print the values
    PROFILER_PRINTING_START;
    print(data);
    PROFILER_PRINTING_STOP;
  } while (continueSimulation);

  deinit(data);

  // De-initialize the runtime library
  runtimeDeinit();

  return 0;
}
