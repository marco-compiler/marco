#ifdef CLI_ENABLE

#include "marco/Runtime/Printers/CSV/CLI.h"
#include "marco/Runtime/Printers/CSV/Options.h"

namespace marco::runtime::printing
{
  std::string CommandLineOptions::getTitle() const
  {
    return "Formatting";
  }

  void CommandLineOptions::printCommandLineOptions(std::ostream& os) const
  {
    os << "  --scientific-notation    Print the values using the scientific notation." << std::endl;
    os << "  --precision=<value>      Set the number of decimals to be printed. Defaults to " << printOptions().precision << "." << std::endl;
  }

  void CommandLineOptions::parseCommandLineOptions(const argh::parser& options) const
  {
    printOptions().scientificNotation = options["scientific-notation"];
    options("precision") >> printOptions().precision;
  }
}

#endif // CLI_ENABLE
