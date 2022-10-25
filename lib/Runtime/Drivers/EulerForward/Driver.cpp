#include "marco/Runtime/Drivers/EulerForward/Driver.h"
#include "marco/Runtime/Drivers/EulerForward/CLI.h"
#include "marco/Runtime/Solvers/EulerForward/Profiler.h"
#include "marco/Runtime/Simulation/Profiler.h"
#include "marco/Runtime/Simulation/Runtime.h"

namespace marco::runtime
{
  EulerForward::EulerForward(Simulation* simulation)
      : Driver(simulation)
  {
  }

  std::unique_ptr<cli::Category> EulerForward::getCLIOptions()
  {
    return std::make_unique<eulerforward::CommandLineOptions>();
  }

  int EulerForward::run()
  {
    getSimulation()->getPrinter()->simulationBegin();

    // Compute the initial conditions.
    EULER_FORWARD_PROFILER_IC_START;
    calcIC(getSimulation()->getData());
    EULER_FORWARD_PROFILER_IC_STOP;

    // Print the initial values.
    getSimulation()->getPrinter()->printValues();

    bool continueSimulation;

    do {
      // Compute the next values of the state variables.
      EULER_FORWARD_PROFILER_STATEVAR_START;
      updateStateVariables(getSimulation()->getData());
      EULER_FORWARD_PROFILER_STATEVAR_STOP;

      // Move to the next step.
      EULER_FORWARD_PROFILER_NONSTATEVAR_START;
      continueSimulation = incrementTime(getSimulation()->getData());
      updateNonStateVariables(getSimulation()->getData());
      EULER_FORWARD_PROFILER_NONSTATEVAR_STOP;

      // Print the values.
      getSimulation()->getPrinter()->printValues();
    } while (continueSimulation);

    getSimulation()->getPrinter()->simulationEnd();

    return 0;
  }
}

namespace marco::runtime
{
  std::unique_ptr<Driver> getDriver(Simulation* simulation)
  {
    return std::make_unique<EulerForward>(simulation);
  }
}
