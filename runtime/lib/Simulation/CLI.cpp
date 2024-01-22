#ifdef CLI_ENABLE

#include "marco/Runtime/Simulation/CLI.h"
#include "marco/Runtime/Simulation/Options.h"
#include <iostream>

namespace marco::runtime::simulation
{
  std::string CommandLineOptions::getTitle() const
  {
    return "General";
  }

  void CommandLineOptions::printCommandLineOptions(
      std::ostream& os) const
  {
    os << "  --start-time=<value>         Set the start time (in seconds)." << std::endl;
    os << "  --end-time=<value>           Set the end time (in seconds)." << std::endl;
    os << "  --thread-equation-chunks     Set the amount of equation chunks each thread would process in a perfectly balanced scenario." << std::endl;
  }

  void CommandLineOptions::parseCommandLineOptions(
      const argh::parser& options) const
  {
    options("start-time") >> getOptions().startTime;
    options("end-time") >> getOptions().endTime;
    options("thread-equation-chunks") >> getOptions().threadEquationChunks;
  }

  std::unique_ptr<cli::Category> getCLIOptions()
  {
    return std::make_unique<CommandLineOptions>();
  }
}

#endif // CLI_ENABLE
