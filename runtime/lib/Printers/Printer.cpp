#include "marco/Runtime/Printers/Printer.h"
#include <cassert>

namespace marco::runtime
{
  Printer::Printer(Simulation* simulation)
      : simulation(simulation)
  {
    assert(simulation != nullptr);
  }

  Printer::~Printer() = default;

  Simulation* Printer::getSimulation()
  {
    return simulation;
  }

  const Simulation* Printer::getSimulation() const
  {
    return simulation;
  }
}
