#include "marco/Options/Options.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/raw_ostream.h"

#define OPTTABLE_VALUES_CODE
#include "marco/Options/Options.inc"
#undef OPTTABLE_VALUES_CODE

#define PREFIX(NAME, VALUE)                                                    \
  static constexpr llvm::StringLiteral NAME##_init[] = VALUE;                  \
  static constexpr llvm::ArrayRef<llvm::StringLiteral> NAME(                   \
      NAME##_init, std::size(NAME##_init) - 1);
#include "marco/Options/Options.inc"
#undef PREFIX

using namespace marco::options;
using namespace llvm::opt;

static constexpr const llvm::StringLiteral PrefixTable_init[] =
#define PREFIX_UNION(VALUES) VALUES
#include "marco/Options/Options.inc"
#undef PREFIX_UNION
    ;
static constexpr const llvm::ArrayRef<llvm::StringLiteral>
    prefixTable(PrefixTable_init, std::size(PrefixTable_init) - 1);

static constexpr OptTable::Info infoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "marco/Options/Options.inc"
#undef OPTION
};

class DriverOptTable : public llvm::opt::PrecomputedOptTable
{
  public:
    DriverOptTable()
        : llvm::opt::PrecomputedOptTable(infoTable, prefixTable)
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
