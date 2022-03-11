#ifndef MARCO_CODEGEN_TRANSFORMS_EXTERNALSOLVER_H
#define MARCO_CODEGEN_TRANSFORMS_EXTERNALSOLVER_H

#include "marco/Codegen/Transforms/Model/Scheduling.h"
#include "marco/Codegen/Transforms/Model/IDA.h"
#include <memory>
#include <vector>

namespace marco::codegen
{
  class ExternalSolver
  {
    public:
      virtual bool isEnabled() const = 0;
  };

  class ExternalSolvers
  {
    public:
      bool containEquation(ScheduledEquation* equation) const;

      void processEquation(ScheduledEquation* equation);

    public:
      IDASolver ida;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_EXTERNALSOLVER_H
