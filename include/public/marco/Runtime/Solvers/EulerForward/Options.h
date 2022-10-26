#ifndef MARCO_RUNTIME_SOLVERS_EULERFORWARD_OPTIONS_H
#define MARCO_RUNTIME_SOLVERS_EULERFORWARD_OPTIONS_H

#include "marco/Runtime/Mangling.h"

namespace marco::runtime::eulerforward
{
  struct Options
  {
    double startTime = 0;
    double endTime = 10;
    double timeStep = 0.1;
  };

  Options& getOptions();
}

#endif // MARCO_RUNTIME_SOLVERS_EULERFORWARD_OPTIONS_H
