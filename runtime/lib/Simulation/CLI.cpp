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
    os << "  --debug                          Enable the debug messages." << std::endl;

    os << "  --start-time=<value>             Set the start time (in seconds)." << std::endl;
    os << "  --end-time=<value>               Set the end time (in seconds)." << std::endl;
    os << "  --equations-partitioning-factor  Set the amount of equation partitions each thread would process in an ideal scenario where all the equations are independent from each other and have equal computational cost." << std::endl;
    os << "  --scheduler-calibration-runs     Set the amount of sequential and multithreaded executions used to decide the execution policy" << std::endl;
  }

  void CommandLineOptions::parseCommandLineOptions(
      const argh::parser& options) const
  {
    getOptions().debug = options["debug"];

    options("start-time") >> getOptions().startTime;
    options("end-time") >> getOptions().endTime;
    options("equations-partitioning-factor") >> getOptions().equationsPartitioningFactor;
    options("scheduler-calibration-runs") >> getOptions().schedulerCalibrationRuns;
  }

  std::unique_ptr<cli::Category> getCLIOptions()
  {
    return std::make_unique<CommandLineOptions>();
  }
}

#endif // CLI_ENABLE
