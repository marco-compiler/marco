#include "marco/Runtime/Drivers/IDA/Driver.h"
#include "marco/Runtime/Drivers/IDA/CLI.h"
#include "marco/Runtime/Solvers/IDA/Profiler.h"
#include "marco/Runtime/Simulation/Profiler.h"
#include "marco/Runtime/Simulation/Runtime.h"

namespace marco::runtime
{
  IDA::IDA(Simulation* simulation)
      : Driver(simulation)
  {
  }

  std::unique_ptr<cli::Category> IDA::getCLIOptions()
  {
    return std::make_unique<ida::CommandLineOptions>();
  }

  int IDA::run()
  {
    getSimulation()->getPrinter()->simulationBegin();
    initICSolvers(getSimulation()->getData());

    // Compute the initial conditions.
    IDA_PROFILER_IC_START;
    calcIC(getSimulation()->getData());
    IDA_PROFILER_IC_STOP;

    deinitICSolvers(getSimulation()->getData());

    // Print the initial values.
    getSimulation()->getPrinter()->printValues();

    initMainSolvers(getSimulation()->getData());

    bool continueSimulation;

    do {
      // Compute the next values of the state variables.
      IDA_PROFILER_STEP_START;
      updateStateVariables(getSimulation()->getData());
      IDA_PROFILER_STEP_STOP;

      // Move to the next step.
      IDA_PROFILER_ALGEBRAIC_VARS_START;
      continueSimulation = incrementTime(getSimulation()->getData());
      updateNonStateVariables(getSimulation()->getData());
      IDA_PROFILER_ALGEBRAIC_VARS_STOP;

      // Print the values.
      getSimulation()->getPrinter()->printValues();
    } while (continueSimulation);

    deinitMainSolvers(getSimulation()->getData());
    getSimulation()->getPrinter()->simulationEnd();

    return 0;
  }
}

namespace marco::runtime
{
  std::unique_ptr<Driver> getDriver(Simulation* simulation)
  {
    return std::make_unique<IDA>(simulation);
  }
}
