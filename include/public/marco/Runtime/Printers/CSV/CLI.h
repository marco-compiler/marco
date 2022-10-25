#ifndef MARCO_RUNTIME_PRINTERS_CSV_CLI_H
#define MARCO_RUNTIME_PRINTERS_CSV_CLI_H

#include "marco/Runtime/CLI/CLI.h"

namespace marco::runtime::printing
{
  class CommandLineOptions : public cli::Category
  {
    std::string getTitle() const override;

    void printCommandLineOptions(std::ostream& os) const override;

    void parseCommandLineOptions(const argh::parser& options) const override;
  };
}

#endif // MARCO_RUNTIME_PRINTERS_CSV_CLI_H
