#include "llvm/ADT/STLExtras.h"
#include "marco/Codegen/Transforms/Model/ExternalSolver.h"

using namespace ::marco;
using namespace ::marco::codegen;
//using namespace ::marco::codegen::modelica;

namespace marco::codegen
{
  bool ExternalSolvers::containEquation(ScheduledEquation* equation) const
  {
    return llvm::any_of(solvers, [equation](const auto& solver) {
      return solver.containsEquation(equation);
    });
  }

  void ExternalSolvers::processEquation(ScheduledEquation* equation)
  {
    for (auto& solver : solvers) {
      if (solver->containsEquation(equation)) {
        solver->processEquation(equation);
      }
    }
  }
}
