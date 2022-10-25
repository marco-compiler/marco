#ifndef MARCO_RUNTIME_DRIVERS_DRIVER_H
#define MARCO_RUNTIME_DRIVERS_DRIVER_H

#include "marco/Runtime/CLI/Category.h"
#include <memory>

namespace marco::runtime
{
  class Simulation;

  class Driver
  {
    public:
      Driver(Simulation* simulation);

      virtual ~Driver();

      /// Get the CLI options to be printed for the selected solver.
      virtual std::unique_ptr<cli::Category> getCLIOptions() = 0;

      virtual int run() = 0;

    protected:
      Simulation* getSimulation();

      const Simulation* getSimulation() const;

    private:
      Simulation* simulation;
  };

  /// Get the driver to be used for the simulation.
  std::unique_ptr<Driver> getDriver(Simulation* simulation);
}

#endif // MARCO_RUNTIME_DRIVERS_DRIVER_H
