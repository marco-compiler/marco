#include "marco/Runtime/Drivers/Driver.h"
#include <cassert>

namespace marco::runtime
{
  Driver::Driver(Simulation* simulation)
      : simulation(simulation)
  {
    assert(simulation != nullptr);
  }

  Driver::~Driver() = default;

  Simulation* Driver::getSimulation()
  {
    return simulation;
  }

  const Simulation* Driver::getSimulation() const
  {
    return simulation;
  }
}
