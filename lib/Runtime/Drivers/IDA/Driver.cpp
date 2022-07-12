#include "marco/Runtime/Drivers/IDA/Driver.h"
#include "marco/Runtime/Drivers/IDA/CLI.h"
#include "marco/Runtime/Solvers/IDA/Options.h"
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
    void* data = getSimulation()->getData();

    // Set the start time.
    setTime(data, ida::getOptions().startTime);

    getSimulation()->getPrinter()->simulationBegin();
    initICSolvers(data);

    // Compute the initial conditions.
    IDA_PROFILER_IC_START;
    calcIC(data);
    IDA_PROFILER_IC_STOP;

    deinitICSolvers(data);

    // Print the initial values.
    getSimulation()->getPrinter()->printValues();

    initMainSolvers(data);

    do {
      // Compute the next values of the state variables.
      IDA_PROFILER_STEP_START;
      updateStateVariables(data);
      IDA_PROFILER_STEP_STOP;

      // Move to the next step.
      IDA_PROFILER_ALGEBRAIC_VARS_START;
      incrementTime(data);
      updateNonStateVariables(data);
      IDA_PROFILER_ALGEBRAIC_VARS_STOP;

      // Print the values.
      getSimulation()->getPrinter()->printValues();
    } while (std::abs(getTime(data) - ida::getOptions().endTime) >=
             ida::getOptions().timeStep);

    deinitMainSolvers(data);
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