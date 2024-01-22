#ifdef SUNDIALS_ENABLE

#include "marco/Runtime/Drivers/IDA/Driver.h"
#include "marco/Runtime/Drivers/IDA/CLI.h"
#include "marco/Runtime/Solvers/IDA/Options.h"
#include "marco/Runtime/Solvers/IDA/Profiler.h"
#include "marco/Runtime/Simulation/Options.h"
#include "marco/Runtime/Simulation/Profiler.h"
#include "marco/Runtime/Simulation/Runtime.h"

//===---------------------------------------------------------------------===//
// Functions defined inside the module of the compiled model
//===---------------------------------------------------------------------===//

extern "C"
{
  /// @name Simulation functions.
  /// {

  /// Compute the initial values of the variables.
  void calcIC();

  /// Compute the values of the variables managed by IDA.
  void updateIDAVariables();

  /// Compute the values of the variables not managed by IDA.
  /// Notice that, from an algebraic point of view, these variables may depend
  /// on the variables within IDA. For this reason, this function should be
  /// called only after 'updateIDAVariables' and after having updated the time.
  void updateNonIDAVariables();

  /// Get the time reached by IDA during the simulation.
  double getIDATime();

  /// }
}

namespace marco::runtime
{
  IDA::IDA(Simulation* simulation)
      : Driver(simulation)
  {
  }

#ifdef CLI_ENABLE
  std::unique_ptr<cli::Category> IDA::getCLIOptions()
  {
    return std::make_unique<sundials::ida::CommandLineOptions>();
  }
#endif // CLI_ENABLE

  int IDA::run()
  {
    // Run the dynamic model.
    calcIC();

    do {
      // Compute the next values of the variables belonging to IDA.
      updateIDAVariables();

      IDA_PROFILER_ALGEBRAIC_VARS_START;
      // Update the time.
      setTime(getIDATime());

      // Compute the next values of the variables not belonging to IDA (which
      // may depend on the IDA variables).
      updateNonIDAVariables();
      IDA_PROFILER_ALGEBRAIC_VARS_STOP;

      // Print the values.
      getSimulation()->getPrinter()->printValues();
    } while (std::abs(getTime() - simulation::getOptions().endTime) >=
             sundials::ida::getOptions().timeStep);

    return EXIT_SUCCESS;
  }
}

namespace marco::runtime
{
  std::unique_ptr<Driver> getDriver(Simulation* simulation)
  {
    return std::make_unique<IDA>(simulation);
  }
}

#endif // SUNDIALS_ENABLE