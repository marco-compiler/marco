#include "marco/codegen/passes/model/Equation.h"
#include "marco/codegen/passes/model/EquationImpl.h"
#include "marco/codegen/passes/model/LoopEquation.h"
#include "marco/codegen/passes/model/ScalarEquation.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

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

  namespace impl
  {
    mlir::LogicalResult explicitate(
        mlir::OpBuilder& builder, modelica::EquationOp equation, const EquationPath& path)
    {
      auto terminator = mlir::cast<EquationSidesOp>(equation.body()->getTerminator());

      for (auto index : path) {
        if (auto status = explicitate(builder, equation, index, path.getEquationSide()); mlir::failed(status)) {
          return status;
        }
      }

      if (path.getEquationSide() == EquationPath::RIGHT) {
        builder.setInsertionPointAfter(terminator);
        builder.create<EquationSidesOp>(terminator->getLoc(), terminator.rhs(), terminator.lhs());
        terminator->erase();
      }

      return mlir::success();
    }

    mlir::LogicalResult explicitate(
        mlir::OpBuilder& builder,
        modelica::EquationOp equation,
        size_t argumentIndex,
        EquationPath::EquationSide side)
    {
      auto terminator = mlir::cast<EquationSidesOp>(equation.body()->getTerminator());
      assert(terminator.lhs().size() == 1);
      assert(terminator.rhs().size() == 1);

      mlir::Value toExplicitate = side == EquationPath::LEFT ? terminator.lhs()[0] : terminator.rhs()[0];
      mlir::Value otherExp = side == EquationPath::RIGHT ? terminator.lhs()[0] : terminator.rhs()[0];

      mlir::Operation* op = toExplicitate.getDefiningOp();

      if (!op->hasTrait<InvertibleOpInterface::Trait>()) {
        return op->emitError("Operation is not invertible");
      }

      return mlir::cast<InvertibleOpInterface>(op).invert(builder, argumentIndex, otherExp);
    }
  }

  mlir::FuncOp BaseEquation::createTemplateFunction(
      mlir::OpBuilder& builder,
      llvm::StringRef functionName,
      mlir::ValueRange vars,
      const EquationPath& path) const
  {
    auto loc = getOperation()->getLoc();
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Create the clone of the equation to be modified
    builder.setInsertionPoint(getOperation());
    auto clonedOp = mlir::cast<EquationOp>(builder.clone(*getOperation().getOperation()));

    if (auto status = impl::explicitate(builder, clonedOp, path); mlir::failed(status)) {
      return nullptr;
    }

    auto module = clonedOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    llvm::SmallVector<mlir::Type, 6> argsTypes;

    // For each iteration variable we need to specify three value: the lower bound, the upper bound
    // and the iteration step.
    argsTypes.append(3 * getNumOfIterationVars(), builder.getIndexType());

    auto varsTypes = vars.getTypes();
    argsTypes.append(varsTypes.begin(), varsTypes.end());

    // Create the "template" function and its entry block
    auto functionType = builder.getFunctionType(argsTypes, llvm::None);
    auto function = builder.create<mlir::FuncOp>(loc, functionName, functionType);

    auto* entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Create the iteration loops
    llvm::SmallVector<mlir::Value, 3> lowerBounds;
    llvm::SmallVector<mlir::Value, 3> upperBounds;
    llvm::SmallVector<mlir::Value, 3> steps;

    for (size_t i = 0, e = getNumOfIterationVars(); i < e; ++i) {
      lowerBounds.push_back(function.getArgument(0 + i * 3));
      upperBounds.push_back(function.getArgument(1 + i * 3));
      steps.push_back(function.getArgument(2 + i * 3));
    }

    std::vector<mlir::Value> iterationVars = createTemplateFunctionLoops(builder, lowerBounds, upperBounds, steps);

    mlir::BlockAndValueMapping mapping;

    // Map the induction variables
    mapIterationVars(mapping, iterationVars);

    // Map the variables
    size_t varsOffset = getNumOfIterationVars() * 3;

    for (size_t i = 0, e = vars.size(); i < e; ++i) {
      mapping.map(vars[i], function.getArgument(i + varsOffset));
    }

    // Clone the equation body
    for (auto& op : clonedOp.body()->getOperations()) {
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

    builder.create<mlir::ReturnOp>(loc);

    // Erase the cloned equation
    clonedOp.erase();

    return function;
  }

  std::vector<mlir::Value> BaseEquation::createTemplateFunctionLoops(
      mlir::OpBuilder& builder,
      mlir::ValueRange lowerBounds,
      mlir::ValueRange upperBounds,
      mlir::ValueRange steps) const
  {
    return {};
  }

  void BaseEquation::mapIterationVars(mlir::BlockAndValueMapping& mapping, mlir::ValueRange iterationVars) const
  {
  }
}
