#ifdef CLI_ENABLE

#include "marco/Runtime/Drivers/EulerForward/CLI.h"
#include "marco/Runtime/Solvers/EulerForward/Options.h"

namespace marco::runtime::eulerforward
{
  std::string CommandLineOptions::getTitle() const
  {
    return "Euler forward";
  }

  void CommandLineOptions::printCommandLineOptions(
      std::ostream& os) const
  {
     os << "  --time-step=<value>    Set the time step (in seconds)." << std::endl;
  }

  void CommandLineOptions::parseCommandLineOptions(
      const argh::parser& options) const
  {
    options("time-step") >> getOptions().timeStep;
  }
}

#endif // CLI_ENABLE
