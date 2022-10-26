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
    os << "  --start-time=<value>                 Set the start time (in seconds)" << std::endl;
    os << "  --end-time=<value>                   Set the end time (in seconds)" << std::endl;
    os << "  --time-step=<value>                  Set the time step (in seconds)" << std::endl;
  }

  void CommandLineOptions::parseCommandLineOptions(
      const argh::parser& options) const
  {
    options("start-time", getOptions().startTime) >> getOptions().startTime;
    options("end-time", getOptions().endTime) >> getOptions().endTime;
    options("time-step", getOptions().timeStep) >> getOptions().timeStep;
  }
}
