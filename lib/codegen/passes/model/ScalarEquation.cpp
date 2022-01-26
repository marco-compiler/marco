#include "marco/codegen/passes/model/ScalarEquation.h"

using namespace ::marco::codegen::modelica;
using namespace ::marco::modeling;

namespace marco::codegen
{
  ScalarEquation::ScalarEquation(EquationOp equation, Variables variables)
      : Impl(equation, variables)
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

  std::unique_ptr<Equation::Impl> ScalarEquation::clone() const
  {
    return std::make_unique<ScalarEquation>(*this);
  }

  std::unique_ptr<Equation::Impl> ScalarEquation::cloneIR() const
  {
    EquationOp equationOp = getOperation();
    mlir::OpBuilder builder(equationOp);
    auto clone = mlir::cast<EquationOp>(builder.clone(*equationOp.getOperation()));
    return std::make_unique<ScalarEquation>(clone, getVariables());
  }

  void ScalarEquation::eraseIR()
  {
    getOperation().erase();
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

  /*
  void ScalarEquation::getWrites(llvm::SmallVectorImpl<ScalarEquation::Access>& accesses) const
  {
    std::vector<Access> result;
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());
    searchAccesses(result, terminator.lhs()[0], EquationPath(EquationPath::LEFT));
    return result;
  }

  void ScalarEquation::getReads(llvm::SmallVectorImpl<ScalarEquation::Access>& accesses) const
  {
    std::vector<Access> result;
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());
    searchAccesses(result, terminator.rhs()[0], EquationPath(EquationPath::LEFT));
    return result;
  }
   */
}
