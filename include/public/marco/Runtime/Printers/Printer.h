#ifndef MARCO_RUNTIME_PRINTERS_PRINTER_H
#define MARCO_RUNTIME_PRINTERS_PRINTER_H

#include "marco/Runtime/CLI/Category.h"
#include <memory>

namespace marco::runtime
{
  class Simulation;

  class Printer
  {
    public:
      Printer(Simulation* simulation);

      virtual ~Printer();

      /// Get the CLI options to be printed for the selected printer.
      virtual std::unique_ptr<cli::Category> getCLIOptions() = 0;

      virtual void simulationBegin() = 0;

      virtual void printValues() = 0;

      virtual void simulationEnd() = 0;

    protected:
      Simulation* getSimulation();

      const Simulation* getSimulation() const;

    private:
      Simulation* simulation;
  };

  std::unique_ptr<Printer> getPrinter(Simulation* simulation);
}

#endif // MARCO_RUNTIME_PRINTERS_PRINTER_H
