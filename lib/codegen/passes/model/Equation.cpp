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

static mlir::LogicalResult removeSubtractions(mlir::OpBuilder& builder, EquationOp equation)
{
  equation.walk([&](SubOp op) {
    mlir::Value rhs = op.rhs();
    rhs = builder.create<NegateOp>(rhs.getLoc(), rhs.getType(), rhs);
    auto addOp = builder.create<AddOp>(rhs.getLoc(), op.resultType(), op.lhs(), rhs);
    op.replaceAllUsesWith(addOp.getOperation());
  });

  return mlir::success();
}

static mlir::LogicalResult distributeMulAndDivOps(mlir::OpBuilder& builder, mlir::Operation* op)
{
  if (op == nullptr) {
    return mlir::success();
  }

  for (auto operand : op->getOperands()) {
    if (auto status = distributeMulAndDivOps(builder, operand.getDefiningOp()); mlir::failed(status)) {
      return status;
    }
  }

  if (auto distributableOp = mlir::dyn_cast<DistributableInterface>(op)) {
    if (!mlir::isa<NegateOp>(op)) {
      mlir::Operation* result = distributableOp.distribute(builder).getDefiningOp();

      if (result != op) {
        op->replaceAllUsesWith(result);
      }
    }
  }

  return mlir::success();
}

static mlir::LogicalResult pushNegateOps(mlir::OpBuilder& builder, mlir::Operation* op)
{
  if (op == nullptr) {
    return mlir::success();
  }

  for (auto operand : op->getOperands()) {
    if (auto status = pushNegateOps(builder, operand.getDefiningOp()); failed(status)) {
      return status;
    }
  }

  if (auto distributableOp = mlir::dyn_cast<NegateOp>(op)) {
    mlir::Operation* result = distributableOp.distribute(builder).getDefiningOp();

    if (result != op) {
      op->replaceAllUsesWith(result);
    }
  }

  return mlir::success();
}

static mlir::LogicalResult flattenSummedValues(mlir::Value value, llvm::SmallVectorImpl<mlir::Value>& values)
{
  if (auto addOp = mlir::dyn_cast<AddOp>(value.getDefiningOp())) {
    if (auto status = flattenSummedValues(addOp.lhs(), values); mlir::failed(status)) {
      return status;
    }

    if (auto status = flattenSummedValues(addOp.rhs(), values); mlir::failed(status)) {
      return status;
    }

    return mlir::success();
  }

  values.emplace_back(value);
  return mlir::success();
}

/*
static bool usesMember(mlir::Value value, AccessToVar access)
{
  if (value == access.getVar())
    return true;

  mlir::Operation* op = value.getDefiningOp();

  if (mlir::isa<LoadOp, SubscriptionOp>(op)) {
    auto subscriptionAccess = AccessToVar::fromExp(Expression::build(value));

    if (access == subscriptionAccess)
      return true;
  }

  if (auto negateOp = mlir::dyn_cast<NegateOp>(op)) {
    if (usesMember(negateOp.operand(), access))
      return true;
  }

  if (auto mulOp = mlir::dyn_cast<MulOp>(op)) {
    if (usesMember(mulOp.lhs(), access) || usesMember(mulOp.rhs(), access))
      return true;
  }

  return false;
}
 */

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

  mlir::Value BaseEquation::getValueAtPath(const EquationPath& path) const
  {
    auto side = path.getEquationSide();
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());
    mlir::Value value = side == EquationPath::LEFT ? terminator.lhs()[0] : terminator.rhs()[0];

    for (auto index : path) {
      mlir::Operation* op = value.getDefiningOp();
      assert(op != nullptr && "Invalid equation path");
      value = op->getOperand(index);
    }

    return value;
  }

  mlir::LogicalResult BaseEquation::explicitate(
      mlir::OpBuilder& builder, const EquationPath& path)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Get all the paths that lead to accesses with the same accessed variable
    // and function.
    auto requestedAccess = getAccessFromPath(path);
    std::vector<Access> accesses;

    for (const auto& access : getAccesses()) {
      if (access.getVariable() == requestedAccess.getVariable() &&
          access.getAccessFunction() == requestedAccess.getAccessFunction()) {
        accesses.push_back(access);
      }
    }

    assert(!accesses.empty());

    // If there is only one access, then it is sufficient to follow the path
    // and invert the operations.

    if (accesses.size() == 1) {
      auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());
      builder.setInsertionPoint(terminator);

      for (auto index : path) {
        if (auto res = explicitate(builder, index, path.getEquationSide()); mlir::failed(res)) {
          return res;
        }
      }

      if (path.getEquationSide() == EquationPath::RIGHT) {
        builder.setInsertionPointAfter(terminator);
        builder.create<EquationSidesOp>(terminator->getLoc(), terminator.rhs(), terminator.lhs());
        terminator->erase();
      }
    }

    // If there are multiple accesses, then we must group all of them and
    // extract the common multiplying factor.

    if (auto res = groupLeftHandSide(builder, *requestedAccess.getVariable(), requestedAccess.getAccessFunction()); mlir::failed(res)) {
      return res;
    }

    return mlir::success();
  }

  std::unique_ptr<Equation> BaseEquation::cloneAndExplicitate(
      mlir::OpBuilder& builder, const EquationPath& path) const
  {
    EquationOp clonedOp = cloneIR();
    auto result = Equation::build(clonedOp, getVariables());

    if (auto res = result->explicitate(builder, path); mlir::failed(res)) {
      result->eraseIR();
      return nullptr;
    }

    return std::move(result);
  }

  mlir::LogicalResult BaseEquation::replaceInto(
      mlir::OpBuilder& builder,
      Equation& destination,
      const ::marco::modeling::AccessFunction& destinationAccessFunction,
      const EquationPath& destinationPath,
      const Access& sourceAccess) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Value destinationValue = destination.getValueAtPath(destinationPath);

    if (destinationValue.getUsers().empty()) {
      // Substitution is useless
      return mlir::success();
    }

    // Determine where the cloned operations will be placed
    mlir::Operation* insertionPoint = destination.getOperation().body()->getTerminator();

    for (const auto& user : destinationValue.getUsers()) {
      if (user->isBeforeInBlock(insertionPoint)) {
        insertionPoint = user;
      }
    }

    builder.setInsertionPoint(insertionPoint);

    // Compose the access
    auto combinedAccess = destinationAccessFunction.combine(sourceAccess.getAccessFunction().inverse());

    // Map the induction variables of the source equation to the destination ones
    mlir::BlockAndValueMapping mapping;

    if (auto res = mapInductionVariables(builder, mapping, destination, combinedAccess); mlir::failed(res)) {
      return res;
    }

    // Obtain the value to be used for the replacement
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());
    mlir::Value replacement = terminator.rhs()[0];

    // The operations to be cloned, in reverse order
    std::vector<mlir::Operation*> toBeCloned;

    // Perform a depth-first traversal of the tree to determine
    // which operations must be cloned and in which order.
    std::stack<mlir::Operation*> stack;

    if (auto op = replacement.getDefiningOp(); op != nullptr) {
      stack.push(op);
    }

    while (!stack.empty()) {
      auto op = stack.top();
      stack.pop();

      toBeCloned.push_back(op);

      for (auto operand : op->getOperands()) {
        if (auto operandOp = operand.getDefiningOp(); operandOp != nullptr) {
          stack.push(operandOp);
        }
      }
    }

    // Clone the operations
    for (auto it = toBeCloned.rbegin(); it != toBeCloned.rend(); ++it) {
      mlir::Operation* op = *it;

      if (auto subscriptionOp = mlir::dyn_cast<SubscriptionOp>(op)) {
        collectSubscriptionIndexes(subscriptionOp.getResult());
      } else if (auto loadOp = mlir::dyn_cast<LoadOp>(op)) {
        collectSubscriptionIndexes(loadOp.getResult());
      } else {
        builder.clone(*op, mapping);
      }
    }

    // Replace the original value with the one obtained through the cloned operations
    destinationValue.replaceAllUsesWith(mapping.lookup(replacement));

    return mlir::success();
  }

  mlir::LogicalResult BaseEquation::explicitate(
      mlir::OpBuilder& builder, size_t argumentIndex, EquationPath::EquationSide side)
  {
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());
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

  mlir::LogicalResult BaseEquation::groupLeftHandSide(
      mlir::OpBuilder& builder, const Variable& variable, const AccessFunction& accessFunction)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    llvm::errs() << "BEFORE GROUPING\n";
    getOperation().dump();

    if (auto status = removeSubtractions(builder, getOperation()); mlir::failed(status)) {
      return status;
    }

    auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());

    if (auto status = distributeMulAndDivOps(builder, terminator.lhs()[0].getDefiningOp()); mlir::failed(status)) {
      return status;
    }

    if (auto status = distributeMulAndDivOps(builder, terminator.rhs()[0].getDefiningOp()); mlir::failed(status)) {
      return status;
    }

    if (auto status = pushNegateOps(builder, terminator.lhs()[0].getDefiningOp()); mlir::failed(status)) {
      return status;
    }

    if (auto status = pushNegateOps(builder, terminator.rhs()[0].getDefiningOp()); mlir::failed(status)) {
      return status;
    }

    llvm::errs() << "AFTER GROUPING\n";
    getOperation().dump();

    llvm::SmallVector<mlir::Value, 3> lhsSummedValues;
    llvm::SmallVector<mlir::Value, 3> rhsSummedValues;

    if (auto status = flattenSummedValues(terminator.rhs()[0], lhsSummedValues); mlir::failed(status)) {
      return status;
    }

    llvm::errs() << "LHS flattened sums\n";

    for (const auto& value : lhsSummedValues) {
      if (auto op = value.getDefiningOp())
        op->dump();
    }

    llvm::errs() << "RHS flattened sums\n";

    for (const auto& value : lhsSummedValues) {
      if (auto op = value.getDefiningOp())
        op->dump();
    }

    if (auto status = flattenSummedValues(terminator.rhs()[0], rhsSummedValues); mlir::failed(status)) {
      return status;
    }

    return mlir::failure();

    /*
    auto* pos = llvm::partition(lhsSummedValues, [&](auto value) {
      return usesMember(value, access);
    });

    builder.setInsertionPoint(equation.getTerminator());

    if (pos == summedValues.begin())
    {
      // There is nothing to be moved to the left-hand side of the equation
      return mlir::success();
    }

    if (pos == summedValues.end())
    {
      // All the right-hand side components should be moved to the left-hand
      // side and thus the variable will take value 0.
      auto terminator = equation.getTerminator();
      mlir::Value zeroValue = builder.create<ConstantOp>(equation.getOp().getLoc(), getIntegerAttribute(builder, terminator.lhs()[0].getType(), 0));
      builder.setInsertionPointAfter(terminator);
      builder.create<EquationSidesOp>(terminator.getLoc(), terminator.lhs(), zeroValue);
      terminator.erase();
      return mlir::success();
    }

    mlir::Value toBeMoved = std::accumulate(
        summedValues.begin(), pos,
        builder.create<ConstantOp>(equation.getOp()->getLoc(), getIntegerAttribute(builder, summedValues[0].getType(), 0)).getResult(),
        [&](mlir::Value acc, mlir::Value value) -> mlir::Value {
          mlir::Value factor = getMultiplyingFactor(builder, value, access);
          return builder.create<AddOp>(value.getLoc(), getMostGenericType(acc.getType(), value.getType()), acc, factor);
        });

    mlir::Value leftFactor = builder.create<SubOp>(
        toBeMoved.getLoc(), toBeMoved.getType(),
        builder.create<ConstantOp>(equation.getOp()->getLoc(), getIntegerAttribute(builder, toBeMoved.getType(), 1)),
        toBeMoved);

    mlir::Value rhs = std::accumulate(
        summedValues.begin() + 1, summedValues.end(),
        builder.create<ConstantOp>(equation.getOp()->getLoc(), getIntegerAttribute(builder, summedValues[0].getType(), 0)).getResult(),
        [&](mlir::Value acc, mlir::Value value) -> mlir::Value {
          return builder.create<AddOp>(value.getLoc(), getMostGenericType(acc.getType(), value.getType()), acc, value);
        });

    rhs = builder.create<DivOp>(
        rhs.getLoc(),
        equation.getTerminator().lhs()[0].getType(),
        rhs, leftFactor);

    auto terminator = equation.getTerminator();
    builder.setInsertionPointAfter(terminator);
    builder.create<EquationSidesOp>(terminator.getLoc(), terminator.lhs(), rhs);
    terminator->erase();
     */

    //return mlir::success();
  }


  std::pair<mlir::Value, std::vector<mlir::Value>> BaseEquation::collectSubscriptionIndexes(mlir::Value value) const
  {
    std::vector<mlir::Value> indexes;
    mlir::Operation* op = value.getDefiningOp();

    while (mlir::isa<LoadOp, SubscriptionOp>(op)) {
      if (auto loadOp = mlir::dyn_cast<LoadOp>(op)) {
        auto loadIndexes = loadOp.indexes();

        for (size_t i = 0, e = loadIndexes.size(); i < e; ++i) {
          indexes.push_back(loadIndexes[e - i - 1]);
        }

        value = loadOp.memory();
      }

      auto subscriptionOp = mlir::cast<SubscriptionOp>(op);
      auto subscriptionIndexes = subscriptionOp.indexes();

      for (size_t i = 0, e = subscriptionIndexes.size(); i < e; ++i) {
        indexes.push_back(subscriptionIndexes[e - i - 1]);
      }

      value = subscriptionOp.source();
    }

    std::reverse(indexes.begin(), indexes.end());
    return std::make_pair(value, std::move(indexes));
  }

  mlir::FuncOp BaseEquation::createTemplateFunction(
      mlir::OpBuilder& builder,
      llvm::StringRef functionName,
      mlir::ValueRange vars,
      ::marco::modeling::scheduling::Direction iterationDirection) const
  {
    auto equation = getOperation();

    auto loc = getOperation()->getLoc();
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto module = equation->getParentOfType<mlir::ModuleOp>();
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

    mlir::BlockAndValueMapping mapping;

    // Map the variables
    size_t varsOffset = getNumOfIterationVars() * 3;

    for (size_t i = 0, e = vars.size(); i < e; ++i) {
      mapping.map(vars[i], function.getArgument(i + varsOffset));
    }

    // Delegate the body creation to the actual equation implementation
    if (auto status = createTemplateFunctionBody(
        builder, mapping, lowerBounds, upperBounds, steps, iterationDirection); mlir::failed(status)) {
      return nullptr;
    }

    builder.create<mlir::ReturnOp>(loc);
    return function;
  }
}
