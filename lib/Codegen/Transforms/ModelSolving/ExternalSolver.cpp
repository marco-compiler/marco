#include "llvm/ADT/STLExtras.h"
#include "marco/Codegen/Transforms/Model/ExternalSolver.h"
#include "marco/Codegen/Transforms/Model/IDA.h"

using namespace ::marco;
using namespace ::marco::codegen;

namespace marco::codegen
{
  ExternalSolver::ExternalSolver(mlir::TypeConverter* typeConverter)
    : typeConverter(typeConverter)
  {
  }

  ExternalSolver::~ExternalSolver() = default;

  mlir::TypeConverter* ExternalSolver::getTypeConverter()
  {
    return typeConverter;
  }

  void ExternalSolvers::addSolver(std::unique_ptr<ExternalSolver> solver)
  {
    solvers.push_back(std::move(solver));
  }

  ExternalSolvers::iterator ExternalSolvers::begin()
  {
    return solvers.begin();
  }

  ExternalSolvers::const_iterator ExternalSolvers::begin() const
  {
    return solvers.begin();
  }

  ExternalSolvers::iterator ExternalSolvers::end()
  {
    return solvers.end();
  }

  ExternalSolvers::const_iterator ExternalSolvers::end() const
  {
    return solvers.end();
  }

  bool ExternalSolvers::containsEquation(ScheduledEquation* equation) const
  {
    return llvm::any_of(solvers, [equation](const auto& solver) {
      return solver->containsEquation(equation);
    });
  }
}
