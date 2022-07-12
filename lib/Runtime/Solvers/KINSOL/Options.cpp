#include "marco/Runtime/Solvers/KINSOL/Options.h"

namespace marco::runtime::kinsol
{
  Options& getOptions()
  {
    static Options options;
    return options;
  }
}
