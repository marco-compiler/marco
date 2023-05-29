#ifndef MARCO_RUNTIME_SUPPORT_CLI_H
#define MARCO_RUNTIME_SUPPORT_CLI_H

#include "marco/Runtime/CLI/CLI.h"

namespace marco::runtime::support
{
  class ApproximationOptions : public cli::Category
  {
    std::string getTitle() const override;

    void printCommandLineOptions(std::ostream& os) const override;

    void parseCommandLineOptions(const argh::parser& options) const override;
  };

  std::unique_ptr<cli::Category> getCLIApproximationOptions();
}

#endif // MARCO_RUNTIME_SUPPORT_CLI_H
