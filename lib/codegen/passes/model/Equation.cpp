#include "marco/codegen/passes/model/Equation.h"
#include "marco/codegen/passes/model/EquationImpl.h"
#include "marco/codegen/passes/model/LoopEquation.h"
#include "marco/codegen/passes/model/ScalarEquation.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::codegen::modelica;
using namespace ::marco::modeling;

static long getIntFromAttribute(mlir::Attribute attribute)
{
  if (auto indexAttr = attribute.dyn_cast<mlir::IntegerAttr>())
    return indexAttr.getInt();

  if (auto booleanAttr = attribute.dyn_cast<BooleanAttribute>())
    return booleanAttr.getValue() ? 1 : 0;

  if (auto integerAttr = attribute.dyn_cast<IntegerAttribute>())
    return integerAttr.getValue();

  if (auto realAttr = attribute.dyn_cast<RealAttribute>())
    return realAttr.getValue();

  assert(false && "Unknown attribute type");
  return 0;
}

/// Check if an equation has explicit or implicit induction variables.
///
/// @param equation  equation
/// @return true if the equation is surrounded by explicit loops or defines implicit ones
static bool hasInductionVariables(EquationOp equation)
{
  auto hasExplicitLoops = [&]() -> bool {
    return equation->getParentOfType<ForEquationOp>() != nullptr;
  };

  auto hasImplicitLoops = [&]() -> bool {
    auto terminator = mlir::cast<EquationSidesOp>(equation.body()->getTerminator());

    return llvm::any_of(terminator.lhs(), [](mlir::Value value) {
      return value.getType().isa<ArrayType>();
    });
  };

  return hasExplicitLoops() || hasImplicitLoops();
}

namespace marco::codegen
{
  std::unique_ptr<Equation> Equation::build(modelica::EquationOp equation, Variables variables)
  {
    if (hasInductionVariables(equation)) {
      return std::make_unique<LoopEquation>(std::move(equation), std::move(variables));
    }

    return std::make_unique<ScalarEquation>(std::move(equation), std::move(variables));
  }

  llvm::Optional<Variable*> Equation::findVariable(mlir::Value value) const
  {
    assert(value.isa<mlir::BlockArgument>());
    const auto& variables = getVariables();

    auto it = llvm::find_if(variables, [&](const std::unique_ptr<Variable>& variable) {
      return value == variable->getValue();
    });

    if (it == variables.end()) {
      return llvm::None;
    }

    return (*it).get();
  }

  bool Equation::isVariable(mlir::Value value) const
  {
    if (value.isa<mlir::BlockArgument>()) {
      return mlir::isa<ModelOp>(value.getParentRegion()->getParentOp());
    }

    return false;
  }

  bool Equation::isReferenceAccess(mlir::Value value) const
  {
    if (isVariable(value))
      return true;

    mlir::Operation* definingOp = value.getDefiningOp();

    if (auto loadOp = mlir::dyn_cast<LoadOp>(definingOp)) {
      return isReferenceAccess(loadOp.memory());
    }

    if (auto viewOp = mlir::dyn_cast<SubscriptionOp>(definingOp)) {
      return isReferenceAccess(viewOp.source());
    }

    return false;
  }

  void Equation::searchAccesses(
      std::vector<Access>& accesses,
      mlir::Value value,
      EquationPath path) const
  {
    std::vector<DimensionAccess> dimensionAccesses;
    searchAccesses(accesses, value, dimensionAccesses, std::move(path));
  }

  void Equation::searchAccesses(
      std::vector<Access>& accesses,
      mlir::Value value,
      std::vector<DimensionAccess>& dimensionAccesses,
      EquationPath path) const
  {
    if (isVariable(value)) {
      resolveAccess(accesses, value, dimensionAccesses, std::move(path));
    } else if (mlir::Operation* definingOp = value.getDefiningOp(); definingOp != nullptr) {
      searchAccesses(accesses, definingOp, dimensionAccesses, std::move(path));
    }
  }

  void Equation::searchAccesses(
      std::vector<Access>& accesses,
      mlir::Operation* op,
      std::vector<DimensionAccess>& dimensionAccesses,
      EquationPath path) const
  {
    auto processIndexesFn = [&](mlir::ValueRange indexes) {
      for (size_t i = 0, e = indexes.size(); i < e; ++i) {
        mlir::Value index = indexes[e - 1 - i];
        auto evaluatedAccess = evaluateDimensionAccess(index);
        dimensionAccesses.push_back(resolveDimensionAccess(evaluatedAccess));
      }
    };

    if (auto loadOp = mlir::dyn_cast<LoadOp>(op)) {
      processIndexesFn(loadOp.indexes());
      searchAccesses(accesses, loadOp.memory(), dimensionAccesses, std::move(path));
    } else if (auto subscriptionOp = mlir::dyn_cast<SubscriptionOp>(op)) {
      processIndexesFn(subscriptionOp.indexes());
      searchAccesses(accesses, subscriptionOp.source(), dimensionAccesses, std::move(path));
    } else {
      for (size_t i = 0, e = op->getNumOperands(); i < e; ++i) {
        EquationPath::Guard guard(path);
        path.append(i);
        searchAccesses(accesses, op->getOperand(i), path);
      }
    }
  }

  void Equation::resolveAccess(
      std::vector<Access>& accesses,
      mlir::Value value,
      std::vector<DimensionAccess>& dimensionsAccesses,
      EquationPath path) const
  {
    auto variable = findVariable(value);

    if (variable.hasValue()) {
      std::vector<DimensionAccess> reverted(dimensionsAccesses.rbegin(), dimensionsAccesses.rend());
      mlir::Type type = value.getType();

      auto arrayType = type.cast<ArrayType>();

      if (arrayType.getRank() == 0)
      {
        assert(dimensionsAccesses.empty());
        reverted.push_back(DimensionAccess::constant(0));
        accesses.emplace_back(*variable, AccessFunction(reverted), std::move(path));
      } else {
        if (arrayType.getShape().size() == dimensionsAccesses.size()) {
          accesses.emplace_back(*variable, AccessFunction(reverted), std::move(path));
        }
      }
    }
  }

  Access Equation::getAccessFromPath(const EquationPath& path) const
  {
    std::vector<Access> accesses;
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());

    auto traverseFn = [&](mlir::Value value, const ExpressionPath& path) -> mlir::Value {
      mlir::Value current = value;

      for (const auto& index : path) {
        mlir::Operation* op = current.getDefiningOp();
        assert(index < op->getNumOperands() && "Invalid expression path");
        current = op->getOperand(index);
      }

      return current;
    };

    if (path.getEquationSide() == EquationPath::LEFT) {
      mlir::Value access = traverseFn(terminator.lhs()[0], path);
      searchAccesses(accesses, access, path);
    } else {
      mlir::Value access = traverseFn(terminator.rhs()[0], path);
      searchAccesses(accesses, access, path);
    }

    assert(accesses.size() == 1);
    return accesses[0];
  }

  std::pair<mlir::Value, long> Equation::evaluateDimensionAccess(mlir::Value value) const
  {
    if (value.isa<mlir::BlockArgument>()) {
      return std::make_pair(value, 0);
    }

    mlir::Operation* op = value.getDefiningOp();
    assert((mlir::isa<ConstantOp>(op) || mlir::isa<AddOp>(op) || mlir::isa<SubOp>(op)) && "Invalid access pattern");

    if (auto constantOp = mlir::dyn_cast<ConstantOp>(op)) {
      return std::make_pair(nullptr, getIntFromAttribute(constantOp.value()));
    }

    if (auto addOp = mlir::dyn_cast<AddOp>(op)) {
      auto first = evaluateDimensionAccess(addOp.lhs());
      auto second = evaluateDimensionAccess(addOp.rhs());

      assert(first.first == nullptr || second.first == nullptr);
      mlir::Value induction = first.first != nullptr ? first.first : second.first;
      return std::make_pair(induction, first.second + second.second);
    }

    auto subOp = mlir::dyn_cast<SubOp>(op);

    auto first = evaluateDimensionAccess(subOp.lhs());
    auto second = evaluateDimensionAccess(subOp.rhs());

    assert(first.first == nullptr || second.first == nullptr);
    mlir::Value induction = first.first != nullptr ? first.first : second.first;
    return std::make_pair(induction, first.second - second.second);
  }

  BaseEquation::BaseEquation(modelica::EquationOp equation, Variables variables)
      : equationOp(equation.getOperation()),
        variables(std::move(variables))
  {
    auto terminator = mlir::cast<EquationSidesOp>(equation.body()->getTerminator());
    assert(terminator.lhs().size() == 1);
    assert(terminator.rhs().size() == 1);
  }

  modelica::EquationOp BaseEquation::getOperation() const
  {
    return mlir::cast<EquationOp>(equationOp);
  }

  const Variables& BaseEquation::getVariables() const
  {
    return variables;
  }

  void BaseEquation::setVariables(Variables value)
  {
    this->variables = std::move(value);
  }

  /*
  mlir::LogicalResult Equation::Impl::explicitate(const EquationPath& path)
  {
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());
    mlir::OpBuilder builder(terminator);

    for (auto index : path)
    {
      if (auto status = explicitate(builder, index, path.getEquationSide()); mlir::failed(status))
        return status;
    }

    if (path.getEquationSide() == EquationPath::RIGHT)
    {
      builder.setInsertionPointAfter(terminator);
      builder.create<EquationSidesOp>(terminator->getLoc(), terminator.rhs(), terminator.lhs());
      terminator->erase();
    }

    return mlir::success();
  }

  mlir::LogicalResult Equation::Impl::explicitate(
      mlir::OpBuilder& builder,
      size_t argumentIndex,
      EquationPath::EquationSide side)
  {
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());
    assert(terminator.lhs().size() == 1);
    assert(terminator.rhs().size() == 1);

    mlir::Value toExplicitate = side == EquationPath::LEFT ? terminator.lhs()[0] : terminator.rhs()[0];
    mlir::Value otherExp = side == EquationPath::RIGHT ? terminator.lhs()[0] : terminator.rhs()[0];

    mlir::Operation* op = toExplicitate.getDefiningOp();

    if (!op->hasTrait<InvertibleOpInterface::Trait>())
      return op->emitError("Operation is not invertible");

    return mlir::cast<InvertibleOpInterface>(op).invert(builder, argumentIndex, otherExp);
  }
   */
}
