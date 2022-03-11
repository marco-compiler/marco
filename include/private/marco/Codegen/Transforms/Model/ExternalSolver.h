#ifndef MARCO_CODEGEN_TRANSFORMS_EXTERNALSOLVER_H
#define MARCO_CODEGEN_TRANSFORMS_EXTERNALSOLVER_H

#include "marco/Codegen/Transforms/Model/Scheduling.h"
#include <memory>
#include <vector>

namespace marco::codegen
{
  class IDASolver;

  class ExternalSolver
  {
    public:
      virtual bool isEnabled() const = 0;
  };

  class ExternalSolvers
  {
    public:
      ExternalSolvers();

      //bool containEquation(ScheduledEquation* equation) const;

      //void processEquation(ScheduledEquation* equation);

    public:
      std::unique_ptr<IDASolver> ida;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_EXTERNALSOLVER_H
