#include "marco/Runtime/Simulation/Options.h"

namespace marco::runtime::simulation
{
  Options& getOptions()
  {
    static Options options;
    return options;
  }
}

