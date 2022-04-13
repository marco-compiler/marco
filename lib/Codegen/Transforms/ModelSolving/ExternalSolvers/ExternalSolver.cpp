#include "marco/Codegen/Transforms/ModelSolving/ExternalSolvers/ExternalSolver.h"
#include "llvm/ADT/STLExtras.h"

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

  size_t ExternalSolvers::size() const
  {
    return solvers.size();
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

  mlir::Value ExternalSolvers::getCurrentTime(mlir::OpBuilder& builder, mlir::ValueRange runtimeDataPtrs) const
  {
    assert(llvm::count_if(solvers, [](const auto& solver) {
      return solver->isEnabled() && solver->hasTimeOwnership();
    }) <= 1);

    for (const auto& solver : llvm::enumerate(solvers)) {
      if (!solver.value()->isEnabled()) {
        continue;
      }

      if (solver.value()->hasTimeOwnership()) {
        return solver.value()->getCurrentTime(builder, runtimeDataPtrs[solver.index()]);
      }
    }

    return nullptr;
  }
}
