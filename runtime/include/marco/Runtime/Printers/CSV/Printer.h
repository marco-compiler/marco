#ifndef MARCO_RUNTIME_PRINTING_PRINTERS_CSV_H
#define MARCO_RUNTIME_PRINTING_PRINTERS_CSV_H

#include "marco/Runtime/Printers/Printer.h"

namespace marco::runtime::printing
{
  class CSVPrinter : public Printer
  {
    public:
      CSVPrinter(Simulation* simulation);

#ifdef CLI_ENABLE
      std::unique_ptr<cli::Category> getCLIOptions() override;
#endif // CLI_ENABLE

      void simulationBegin() override;

      void printValues() override;

      void simulationEnd() override;
  };
}

#endif // MARCO_RUNTIME_PRINTING_PRINTERS_CSV_H
