#ifndef MARCO_RUNTIME_SIMULATION_CLI_H
#define MARCO_RUNTIME_SIMULATION_CLI_H

#ifdef CLI_ENABLE

#include "marco/Runtime/CLI/CLI.h"

namespace marco::runtime::simulation
{
  class CommandLineOptions : public cli::Category
  {
    public:
      std::string getTitle() const override;

      void printCommandLineOptions(std::ostream& os) const override;

      void parseCommandLineOptions(const argh::parser& options) const override;
  };

#ifdef CLI_ENABLE
  std::unique_ptr<cli::Category> getCLIOptions();
#endif // CLI_ENABLE
}

#endif // CLI_ENABLE

#endif // MARCO_RUNTIME_SIMULATION_CLI_H
