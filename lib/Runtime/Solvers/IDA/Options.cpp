#include "marco/Runtime/Solvers/IDA/Options.h"

namespace marco::runtime::ida
{
  Options& getOptions()
  {
    static Options options;
    return options;
  }
}
