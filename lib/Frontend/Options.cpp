#include "marco/Frontend/Options.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include <cassert>

#define PREFIX(NAME, VALUE) static const char *const NAME[] = VALUE;
#include "marco/Frontend/Options.inc"
#undef PREFIX

using namespace marco::frontend;
using namespace marco::frontend::options;
using namespace llvm::opt;

static const llvm::opt::OptTable::Info InfoTable[] = {
    #define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
                  HELPTEXT, METAVAR, VALUES)                                      \
     {PREFIX, NAME,  HELPTEXT,    METAVAR,     OPT_##ID,  Option::KIND##Class,    \
      PARAM,  FLAGS, OPT_##GROUP, OPT_##ALIAS, ALIASARGS, VALUES},
    #include "marco/Frontend/Options.inc"
    #undef OPTION
};

class DriverOptTable : public llvm::opt::OptTable {
  public:
    DriverOptTable()
        : llvm::opt::OptTable(InfoTable)
    {
    }
};

namespace marco::frontend
{
  const llvm::opt::OptTable& getDriverOptTable() {
    static const DriverOptTable *Table = []() {
      auto Result = std::make_unique<DriverOptTable>();
      // Options.inc is included in DriverOptions.cpp, and calls OptTable's addValues function.
      // Opt is a variable used in the code fragment in Options.inc.

      llvm::opt::OptTable &Opt = *Result;

      #define OPTTABLE_ARG_INIT
      #include "marco/Frontend/Options.inc"
      #undef OPTTABLE_ARG_INIT

      return Result.release();
    }();

    return *Table;
  }
}
