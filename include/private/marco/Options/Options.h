#ifndef MARCO_DRIVER_OPTIONS_H
#define MARCO_DRIVER_OPTIONS_H

#include "llvm/Option/Option.h"

namespace marco::options
{
  enum MARCOFlags {
    DriverOption = (1 << 10),
    MC1Option = (1 << 11)
  };

  enum ID {
    OPT_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, \
               VISIBILITY, PARAM, HELPTEXT, METAVAR, VALUES)           \
    OPT_##ID,
#include "marco/Options/Options.inc"
    LastOption
#undef OPTION
  };

  const llvm::opt::OptTable& getDriverOptTable();

  void printHelp();
}

#endif // MARCO_DRIVER_OPTIONS_H
