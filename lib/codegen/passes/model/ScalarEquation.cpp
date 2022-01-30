#include "marco/codegen/passes/model/ScalarEquation.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace ::marco::codegen::modelica;
using namespace ::marco::modeling;

namespace marco::codegen
{
  ScalarEquation::ScalarEquation(EquationOp equation, Variables variables)
      : BaseEquation(equation, variables)
  {
    // Check that the equation is not enclosed in a loop
    assert(equation->getParentOfType<ForEquationOp>() == nullptr);

    // Check that all the values are scalars
    auto terminator = mlir::cast<EquationSidesOp>(equation.body()->getTerminator());

    auto isScalarFn = [](mlir::Value value) {
      auto type = value.getType();
      return type.isa<BooleanType>() || type.isa<IntegerType>() || type.isa<RealType>();
    };

    assert(llvm::all_of(terminator.lhs(), isScalarFn));
    assert(llvm::all_of(terminator.rhs(), isScalarFn));
  }

  std::unique_ptr<Equation> ScalarEquation::clone() const
  {
    return std::make_unique<ScalarEquation>(*this);
  }

  size_t ScalarEquation::getNumOfIterationVars() const
  {
    return 1;
  }

  long ScalarEquation::getRangeBegin(size_t inductionVarIndex) const
  {
    return 0;
  }

  long ScalarEquation::getRangeEnd(size_t inductionVarIndex) const
  {
    return 1;
  }

  std::vector<Access> ScalarEquation::getAccesses() const
  {
    std::vector<Access> accesses;
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());

    auto processFn = [&](mlir::Value value, EquationPath path) {
      searchAccesses(accesses, value, std::move(path));
    };

    processFn(terminator.lhs()[0], EquationPath(EquationPath::LEFT));
    processFn(terminator.rhs()[0], EquationPath(EquationPath::RIGHT));

    return accesses;
  }

  DimensionAccess ScalarEquation::resolveDimensionAccess(std::pair<mlir::Value, long> access) const
  {
    assert(access.first == nullptr);
    return DimensionAccess::constant(access.second);
  }
}
