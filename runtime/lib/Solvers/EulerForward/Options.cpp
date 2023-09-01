#include "marco/Runtime/Solvers/EulerForward/Options.h"

using namespace ::marco::runtime::eulerforward;

namespace marco::runtime::eulerforward
{
  Options& getOptions()
  {
    static Options options;
    return options;
  }
}
