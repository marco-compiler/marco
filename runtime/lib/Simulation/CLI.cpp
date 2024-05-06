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
    os << "  --debug                      Enable the debug messages." << std::endl;

    os << "  --start-time=<value>         Set the start time (in seconds)." << std::endl;
    os << "  --end-time=<value>           Set the end time (in seconds)." << std::endl;
    os << "  --equations-chunks-factor    Set the amount of equation chunks each thread would process in a perfectly balanced scenario." << std::endl;
  }

  void CommandLineOptions::parseCommandLineOptions(
      const argh::parser& options) const
  {
    getOptions().debug = options["debug"];

    options("start-time") >> getOptions().startTime;
    options("end-time") >> getOptions().endTime;
    options("equations-chunks-factor") >> getOptions().equationsChunksFactor;
  }

  std::unique_ptr<cli::Category> getCLIOptions()
  {
    return std::make_unique<CommandLineOptions>();
  }
}

#endif // CLI_ENABLE
