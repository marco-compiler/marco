#include "llvm/ADT/STLExtras.h"
#include "marco/Codegen/Transforms/Model/ExternalSolver.h"
#include "marco/Codegen/Transforms/Model/IDA.h"

using namespace ::marco;
using namespace ::marco::codegen;
//using namespace ::marco::codegen::modelica;

namespace marco::codegen
{
  ExternalSolvers::ExternalSolvers() : ida(std::make_unique<IDASolver>())
  {
  }

  /*
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
   */
}
