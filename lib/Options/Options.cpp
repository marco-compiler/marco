#include "marco/Options/Options.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

#define PREFIX(NAME, VALUE) static const char *const NAME[] = VALUE;
#include "marco/Options/Options.inc"
#undef PREFIX

using namespace marco::options;
using namespace llvm::opt;

static const llvm::opt::OptTable::Info infoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,     \
                  HELPTEXT, METAVAR, VALUES)                                      \
     {PREFIX, NAME,  HELPTEXT,    METAVAR,     OPT_##ID,  Option::KIND##Class,    \
      PARAM,  FLAGS, OPT_##GROUP, OPT_##ALIAS, ALIASARGS, VALUES},
#include "marco/Options/Options.inc"
#undef OPTION
};

class DriverOptTable : public llvm::opt::OptTable
{
  public:
    DriverOptTable()
        : llvm::opt::OptTable(infoTable)
    {
    }
};

namespace marco::options
{
  const llvm::opt::OptTable& getDriverOptTable()
  {
    static const DriverOptTable* table = []() {
      auto result = std::make_unique<DriverOptTable>();
      // Options.inc is included in DriverOptions.cpp, and calls OptTable's addValues function.
      // Opt is a variable used in the code fragment in Options.inc.

      [[maybe_unused]] llvm::opt::OptTable& Opt = *result;

#define OPTTABLE_ARG_INIT
#include "marco/Options/Options.inc"
#undef OPTTABLE_ARG_INIT

      return result.release();
    }();

    return *table;
  }

  void printHelp()
  {
    llvm::outs() << "MARCO - Modelica Advanced Research COmpiler\n";
    llvm::outs() << "Website: https://github.com/modelica-polimi/marco\n";
  }
}
