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
    os << "  --scientific-notation      Print the values using the scientific notation.\n";
    os << "  --precision=<value>        Set the number of decimals to be printed.\n";
  }

  void CommandLineOptions::parseCommandLineOptions(const argh::parser& options) const
  {
    printOptions().scientificNotation = options["scientific-notation"];
    options("precision", printOptions().precision) >> printOptions().precision;
  }
}
