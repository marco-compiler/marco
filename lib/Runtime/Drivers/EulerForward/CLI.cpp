#include "marco/Runtime/Drivers/EulerForward/CLI.h"

namespace marco::runtime::eulerforward
{
  std::string CommandLineOptions::getTitle() const
  {
    return "Euler forward";
  }

  void CommandLineOptions::printCommandLineOptions(std::ostream& os) const
  {
  }

  void CommandLineOptions::parseCommandLineOptions(
      const argh::parser& options) const
  {
  }
}
