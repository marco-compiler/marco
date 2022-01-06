#ifndef MARCO_FRONTEND_OPTIONS_H
#define MARCO_FRONTEND_OPTIONS_H

#include <llvm/Option/Option.h>

namespace marco::frontend
{
  //#define ORIGINAL 1

  namespace options
  {
    #ifdef ORIGINAL
    enum ClangFlags {
      NoXarchOption = (1 << 4),
      LinkerInput = (1 << 5),
      NoArgumentUnused = (1 << 6),
      Unsupported = (1 << 7),
      CoreOption = (1 << 8),
      CLOption = (1 << 9),
      CC1Option = (1 << 10),
      CC1AsOption = (1 << 11),
      NoDriverOption = (1 << 12),
      LinkOption = (1 << 13),
      FlangOption = (1 << 14),
      FC1Option = (1 << 15),
      FlangOnlyOption = (1 << 16),
      Ignored = (1 << 17),
    };
    #else
    enum MARCOFlags {
      MC1Option = (1 << 4),
    };
    #endif

    enum ID {
      OPT_INVALID = 0, // This is not an option ID.
      #define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
      OPT_##ID,
      #include <marco/frontend/Options.inc>
      LastOption
      #undef OPTION
    };
  }

  const llvm::opt::OptTable& getDriverOptTable();
}

#endif // MARCO_FRONTEND_OPTIONS_H
