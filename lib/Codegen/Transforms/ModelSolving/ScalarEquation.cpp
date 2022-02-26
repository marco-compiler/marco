#include "marco/Codegen/Transforms/Model/ScalarEquation.h"
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
    [[maybe_unused]] auto isScalarFn = [](mlir::Value value) {
      auto type = value.getType();
      return type.isa<BooleanType>() || type.isa<IntegerType>() || type.isa<RealType>();
    };

    assert(llvm::all_of(getTerminator().lhs(), isScalarFn));
    assert(llvm::all_of(getTerminator().rhs(), isScalarFn));
  }

  std::unique_ptr<Equation> ScalarEquation::clone() const
  {
    return std::make_unique<ScalarEquation>(*this);
  }

  EquationOp ScalarEquation::cloneIR() const
  {
    EquationOp equationOp = getOperation();
    mlir::OpBuilder builder(equationOp);
    return mlir::cast<EquationOp>(builder.clone(*equationOp.getOperation()));
  }

  void ScalarEquation::eraseIR()
  {
    getOperation().erase();
  }

  void ScalarEquation::dumpIR() const
  {
    getOperation().dump();
  }

  size_t ScalarEquation::getNumOfIterationVars() const
  {
    return 1;
  }

  MultidimensionalRange ScalarEquation::getIterationRanges() const
  {
    return MultidimensionalRange(Point(0));
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

  std::vector<mlir::Value> ScalarEquation::getInductionVariables() const
  {
    return {};
  }

  mlir::LogicalResult ScalarEquation::mapInductionVariables(
      mlir::OpBuilder& builder,
      mlir::BlockAndValueMapping& mapping,
      Equation& destination,
      const ::marco::modeling::AccessFunction& transformation) const
  {
    // Nothing to be mapped
    return mlir::success();
  }

  mlir::LogicalResult ScalarEquation::createTemplateFunctionBody(
      mlir::OpBuilder& builder,
      mlir::BlockAndValueMapping& mapping,
      mlir::ValueRange beginIndexes,
      mlir::ValueRange endIndexes,
      mlir::ValueRange steps,
      ::marco::modeling::scheduling::Direction iterationDirection) const
  {
    auto equation = getOperation();
    auto loc = equation.getLoc();

    for (auto& op : equation.body()->getOperations()) {
      if (auto terminator = mlir::dyn_cast<modelica::EquationSidesOp>(op)) {
        // Convert the equality into an assignment
        for (auto [lhs, rhs] : llvm::zip(terminator.lhs(), terminator.rhs())) {
          auto mappedLhs = mapping.lookup(lhs);
          auto mappedRhs = mapping.lookup(rhs);

          if (auto loadOp = mlir::dyn_cast<LoadOp>(mappedLhs.getDefiningOp())) {
            assert(loadOp.indexes().empty());
            builder.create<AssignmentOp>(loc, mappedRhs, loadOp.memory());
          } else {
            builder.create<AssignmentOp>(loc, mappedRhs, mappedLhs);
          }
        }
      } else {
        // Clone all the other operations
        builder.clone(op, mapping);
      }
    }

    return mlir::success();
  }
}
