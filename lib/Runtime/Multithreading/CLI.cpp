#include "marco/Runtime/Multithreading/CLI.h"
#include "marco/Runtime/Multithreading/Options.h"
#include <iostream>

namespace marco::runtime::multithreading
{
  std::string CommandLineOptions::getTitle() const
  {
    return "Multithreading";
  }

  void CommandLineOptions::printCommandLineOptions(
      std::ostream& os) const
  {
    os << "  --disable-multithreading     Disable the usage of multiple threads." << std::endl;
    os << "  --threads=<value>            Set the amount of threads to be used." << std::endl;
  }

  void CommandLineOptions::parseCommandLineOptions(
      const argh::parser& options) const
  {
    multithreadingOptions().enableMultithreading = !options["disable-multithreading"];
    options("threads") >> multithreadingOptions().numOfThreads;
  }

  std::unique_ptr<cli::Category> getCLIOptions()
  {
    return std::make_unique<CommandLineOptions>();
  }
}
