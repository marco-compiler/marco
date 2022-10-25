#ifndef MARCO_RUNTIME_DRIVERS_EULERFORWARD_CLI_H
#define MARCO_RUNTIME_DRIVERS_EULERFORWARD_CLI_H

#include "marco/Runtime/CLI/CLI.h"

namespace marco::runtime::eulerforward
{
  class CommandLineOptions : public cli::Category
  {
    public:
      std::string getTitle() const override;

      void printCommandLineOptions(std::ostream& os) const override;

      void parseCommandLineOptions(const argh::parser& options) const override;
  };
}

#endif // MARCO_RUNTIME_DRIVERS_EULERFORWARD_CLI_H
