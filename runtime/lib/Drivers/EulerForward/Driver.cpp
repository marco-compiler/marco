#include "marco/Runtime/Drivers/EulerForward/Driver.h"
#include "marco/Runtime/Drivers/EulerForward/CLI.h"
#include "marco/Runtime/Solvers/EulerForward/Options.h"
#include "marco/Runtime/Solvers/EulerForward/Profiler.h"
#include "marco/Runtime/Simulation/Options.h"
#include "marco/Runtime/Simulation/Profiler.h"
#include "marco/Runtime/Simulation/Runtime.h"
#include <iostream>

namespace marco::runtime
{
  EulerForward::EulerForward(Simulation* simulation)
      : Driver(simulation)
  {
  }

#ifdef CLI_ENABLE
  std::unique_ptr<cli::Category> EulerForward::getCLIOptions()
  {
    return std::make_unique<eulerforward::CommandLineOptions>();
  }
#endif // CLI_ENABLE

  int EulerForward::run()
  {
    if (marco::runtime::simulation::getOptions().debug) {
      std::cerr << "[Euler Forward] Starting simulation" << std::endl;
    }

    double time;

    do {
      // Compute the next values of the state variables.
      if (marco::runtime::simulation::getOptions().debug) {
        std::cerr << "[Euler Forward] Updating state variables" << std::endl;
      }

      EULER_FORWARD_PROFILER_STATEVAR_START;
      updateStateVariables(eulerforward::getOptions().timeStep);
      EULER_FORWARD_PROFILER_STATEVAR_STOP;

      // Move to the next step.
      if (marco::runtime::simulation::getOptions().debug) {
        std::cerr << "[Euler Forward] Updating time and non-state variables" << std::endl;
      }

      EULER_FORWARD_PROFILER_NONSTATEVAR_START;
      time = getTime() + eulerforward::getOptions().timeStep;
      setTime(time);

      updateNonStateVariables();
      EULER_FORWARD_PROFILER_NONSTATEVAR_STOP;

      if (marco::runtime::simulation::getOptions().debug) {
        std::cerr << "[Euler Forward] Printing values" << std::endl;
      }

      // Print the values.
      getSimulation()->getPrinter()->printValues();
    } while (std::abs(simulation::getOptions().endTime - time) >=
             eulerforward::getOptions().timeStep);

    if (marco::runtime::simulation::getOptions().debug) {
      std::cerr << "[Euler Forward] Simulation finished" << std::endl;
    }

    return EXIT_SUCCESS;
  }
}

namespace marco::runtime
{
  std::unique_ptr<Driver> getDriver(Simulation* simulation)
  {
    return std::make_unique<EulerForward>(simulation);
  }
}
