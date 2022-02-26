#ifndef MARCO_FRONTEND_OPTIONS_H
#define MARCO_FRONTEND_OPTIONS_H

#include "llvm/Option/Option.h"

namespace marco::frontend
{
  namespace options
  {
    enum MARCOFlags {
      MC1Option = (1 << 4),
    };

    enum ID {
      OPT_INVALID = 0, // This is not an option ID.
      #define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
      OPT_##ID,
      #include "marco/Frontend/Options.inc"
      LastOption
      #undef OPTION
    };
  }

  const llvm::opt::OptTable& getDriverOptTable();
}

#endif // MARCO_FRONTEND_OPTIONS_H
