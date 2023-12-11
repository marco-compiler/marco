#ifdef SUNDIALS_ENABLE

#include "marco/Runtime/Drivers/IDA/Driver.h"
#include "marco/Runtime/Drivers/IDA/CLI.h"
#include "marco/Runtime/Solvers/IDA/Options.h"
#include "marco/Runtime/Solvers/IDA/Profiler.h"
#include "marco/Runtime/Simulation/Profiler.h"
#include "marco/Runtime/Simulation/Runtime.h"

//===---------------------------------------------------------------------===//
// Functions defined inside the module of the compiled model
//===---------------------------------------------------------------------===//

extern "C"
{
  /// @name Simulation functions.
  /// {

  /// Initialize the solvers to be used for 'initial conditions' model.
  void* initICSolvers();

  /// Deinitialize the solver used for the 'initial conditions' model.
  void* deinitICSolvers();

  /// Solve the model for the initial conditions.
  void solveICModel();

  /// Initialize the solvers to be used for 'main' model.
  void initMainSolvers();

  /// Deinitialize the solver used for the 'main' model.
  void deinitMainSolvers();

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
    return std::make_unique<ida::CommandLineOptions>();
  }
#endif // CLI_ENABLE

  int IDA::run()
  {
    // Set the start time.
    setTime(ida::getOptions().startTime);

    getSimulation()->getPrinter()->simulationBegin();

    // Process the "initial conditions model".
    initICSolvers();
    solveICModel();
    deinitICSolvers();

    // Print the initial values.
    getSimulation()->getPrinter()->printValues();

    // Process the "main model".
    initMainSolvers();
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
    } while (std::abs(getTime() - ida::getOptions().endTime) >=
             ida::getOptions().timeStep);

    // Deinitialize the simulation.
    deinitMainSolvers();
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

#endif // SUNDIALS_ENABLE