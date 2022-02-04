#include "marco/codegen/passes/model/Equation.h"
#include "marco/codegen/passes/model/EquationImpl.h"
#include "marco/codegen/passes/model/LoopEquation.h"
#include "marco/codegen/passes/model/ScalarEquation.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <numeric>

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

  llvm_unreachable("Unknown attribute type");
  return 0;
}

static mlir::Attribute getIntegerAttribute(mlir::OpBuilder& builder, mlir::Type type, int value)
{
  if (type.isa<BooleanType>()) {
    return BooleanAttribute::get(type, value > 0);
  }

  if (type.isa<IntegerType>()) {
    return IntegerAttribute::get(type, value);
  }

  if (type.isa<RealType>()) {
    return RealAttribute::get(type, value);
  }

  return builder.getIndexAttr(value);
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

static std::pair<mlir::Value, std::vector<mlir::Value>> collectSubscriptionIndexes(mlir::Value value)
{
  std::vector<mlir::Value> indexes;
  mlir::Operation* op = value.getDefiningOp();

  while (op != nullptr && mlir::isa<LoadOp, SubscriptionOp>(op)) {
    if (auto loadOp = mlir::dyn_cast<LoadOp>(op)) {
      auto loadIndexes = loadOp.indexes();

      for (size_t i = 0, e = loadIndexes.size(); i < e; ++i) {
        indexes.push_back(loadIndexes[e - i - 1]);
      }

      value = loadOp.memory();
      op = value.getDefiningOp();
    } else {
      auto subscriptionOp = mlir::cast<SubscriptionOp>(op);
      auto subscriptionIndexes = subscriptionOp.indexes();

      for (size_t i = 0, e = subscriptionIndexes.size(); i < e; ++i) {
        indexes.push_back(subscriptionIndexes[e - i - 1]);
      }

      value = subscriptionOp.source();
      op = value.getDefiningOp();
    }
  }

  std::reverse(indexes.begin(), indexes.end());
  return std::make_pair(value, std::move(indexes));
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
    if (auto res = distributeMulAndDivOps(builder, operand.getDefiningOp()); mlir::failed(res)) {
      return res;
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
    if (auto res = pushNegateOps(builder, operand.getDefiningOp()); mlir::failed(res)) {
      return res;
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

static mlir::LogicalResult collectSummedValues(std::vector<mlir::Value>& result, mlir::Value root)
{
  if (auto addOp = mlir::dyn_cast<AddOp>(root.getDefiningOp())) {
    if (auto res = collectSummedValues(result, addOp.lhs()); mlir::failed(res)) {
      return res;
    }

    if (auto res = collectSummedValues(result, addOp.rhs()); mlir::failed(res)) {
      return res;
    }

    return mlir::success();
  }

  result.push_back(root);
  return mlir::success();
}

static mlir::Type getMostGenericType(mlir::Type x, mlir::Type y)
{
  if (x.isa<BooleanType>()) {
    return y;
  }

  if (y.isa<BooleanType>()) {
    return x;
  }

  if (x.isa<RealType>()) {
    return x;
  }

  if (y.isa<RealType>()) {
    return y;
  }

  if (x.isa<IntegerType>()) {
    return x;
  }

  return y;
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

      if (arrayType.getRank() == 0) {
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

    if (auto res = groupLeftHandSide(builder, requestedAccess); mlir::failed(res)) {
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
    std::stack<mlir::Operation*> cloneStack;

    if (auto op = replacement.getDefiningOp(); op != nullptr) {
      cloneStack.push(op);
    }

    while (!cloneStack.empty()) {
      auto op = cloneStack.top();
      cloneStack.pop();

      toBeCloned.push_back(op);

      for (const auto& operand : op->getOperands()) {
        if (auto operandOp = operand.getDefiningOp(); operandOp != nullptr) {
          cloneStack.push(operandOp);
        }
      }
    }

    // Clone the operations
    for (auto it = toBeCloned.rbegin(); it != toBeCloned.rend(); ++it) {
      mlir::Operation* op = *it;
      builder.clone(*op, mapping);
    }

    // Replace the original value with the one obtained through the cloned operations
    destinationValue.replaceAllUsesWith(mapping.lookup(replacement));

    // Erase the replaced operations, which are now useless
    std::stack<mlir::Operation*> eraseStack;

    if (auto op = destinationValue.getDefiningOp(); op != nullptr) {
      eraseStack.push(op);
    }

    while (!eraseStack.empty()) {
      auto op = eraseStack.top();
      eraseStack.pop();

      if (op->getUsers().empty()) {
        for (const auto& operand : op->getOperands()) {
          if (auto operandOp = operand.getDefiningOp(); operandOp != nullptr) {
            eraseStack.push(operandOp);
          }
        }

        op->erase();
      }
    }

    return mlir::success();
  }

  EquationSidesOp BaseEquation::getTerminator() const
  {
    return mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());
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
      mlir::OpBuilder& builder, const Access& access)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Determine whether the access to be grouped is inside both the equation's sides or just one of them.
    // When the requested access is found, also check that the path goes through linear operations. If not,
    // explicitation is not possible.
    bool lhsHasAccess = false;
    bool rhsHasAccess = false;

    for (const auto& acc : getAccesses()) {
      if (acc.getVariable() == access.getVariable() && acc.getAccessFunction() == access.getAccessFunction()) {
        lhsHasAccess |= acc.getPath().getEquationSide() == EquationPath::LEFT;
        rhsHasAccess |= acc.getPath().getEquationSide() == EquationPath::RIGHT;

        // TODO check linearity
      }
    }

    if (auto res = removeSubtractions(builder, getOperation()); mlir::failed(res)) {
      return res;
    }

    auto convertToSumsFn = [&](std::function<mlir::Value()> root) -> mlir::LogicalResult {
      if (auto res = distributeMulAndDivOps(builder, root().getDefiningOp()); mlir::failed(res)) {
        return res;
      }

      if (auto res = pushNegateOps(builder, root().getDefiningOp()); mlir::failed(res)) {
        return res;
      }

      return mlir::success();
    };

    std::vector<mlir::Value> lhsSummedValues;
    std::vector<mlir::Value> rhsSummedValues;

    if (lhsHasAccess) {
      auto rootFn = [&]() -> mlir::Value {
        return getTerminator().lhs()[0];
      };

      if (auto res = convertToSumsFn(rootFn); mlir::failed(res)) {
        return res;
      }

      if (auto res = collectSummedValues(lhsSummedValues, rootFn()); mlir::failed(res)) {
        return res;
      }
    }

    if (rhsHasAccess) {
      auto rootFn = [&]() -> mlir::Value {
        return getTerminator().rhs()[0];
      };

      if (auto res = convertToSumsFn(rootFn); mlir::failed(res)) {
        return res;
      }

      if (auto res = collectSummedValues(rhsSummedValues, rootFn()); mlir::failed(res)) {
        return res;
      }
    }

    auto containsAccessFn = [&](mlir::Value value, const Access& access, EquationPath::EquationSide side) -> bool {
      EquationPath path(side);
      std::vector<Access> accesses;
      searchAccesses(accesses, value, path);

      return llvm::any_of(accesses, [&](const Access& acc) {
        return acc.getVariable() == access.getVariable() && acc.getAccessFunction() == access.getAccessFunction();
      });
    };

    builder.setInsertionPoint(getTerminator());

    if (lhsHasAccess && rhsHasAccess) {
      auto leftPos = llvm::partition(lhsSummedValues, [&](const auto& value) {
        return containsAccessFn(value, access, EquationPath::LEFT);
      });

      auto rightPos = llvm::partition(rhsSummedValues, [&](const auto& value) {
        return containsAccessFn(value, access, EquationPath::RIGHT);
      });

      mlir::Value lhsFactor = std::accumulate(
          lhsSummedValues.begin(), leftPos,
          builder.create<ConstantOp>(getOperation()->getLoc(), RealAttribute::get(builder.getContext(), 0)).getResult(),
          [&](mlir::Value acc, mlir::Value value) -> mlir::Value {
            mlir::Value factor = getMultiplyingFactor(builder, value, access.getVariable()->getValue(), access.getAccessFunction());
            return builder.create<AddOp>(value.getLoc(), getMostGenericType(acc.getType(), value.getType()), acc, factor);
          });

      mlir::Value rhsFactor = std::accumulate(
          rhsSummedValues.begin(), rightPos,
          builder.create<ConstantOp>(getOperation()->getLoc(), RealAttribute::get(builder.getContext(), 0)).getResult(),
          [&](mlir::Value acc, mlir::Value value) -> mlir::Value {
            mlir::Value factor = getMultiplyingFactor(builder, value, access.getVariable()->getValue(), access.getAccessFunction());
            return builder.create<AddOp>(value.getLoc(), getMostGenericType(acc.getType(), value.getType()), acc, factor);
          });

      mlir::Value lhsRemaining = std::accumulate(
          leftPos, lhsSummedValues.end(),
          builder.create<ConstantOp>(getOperation()->getLoc(), RealAttribute::get(builder.getContext(), 0)).getResult(),
          [&](mlir::Value acc, mlir::Value value) -> mlir::Value {
            return builder.create<AddOp>(value.getLoc(), getMostGenericType(acc.getType(), value.getType()), acc, value);
          });

      mlir::Value rhsRemaining = std::accumulate(
          rightPos, rhsSummedValues.end(),
          builder.create<ConstantOp>(getOperation()->getLoc(), RealAttribute::get(builder.getContext(), 0)).getResult(),
          [&](mlir::Value acc, mlir::Value value) -> mlir::Value {
            return builder.create<AddOp>(value.getLoc(), getMostGenericType(acc.getType(), value.getType()), acc, value);
          });

      auto terminator = getTerminator();
      auto loc = terminator->getLoc();

      auto lhs = getValueAtPath(access.getPath());

      mlir::Value rhs = builder.create<DivOp>(
          loc, lhs.getType(),
          builder.create<SubOp>(loc, getMostGenericType(rhsRemaining.getType(), lhsRemaining.getType()), rhsRemaining, lhsRemaining),
          builder.create<SubOp>(loc, getMostGenericType(lhsFactor.getType(), rhsFactor.getType()), lhsFactor, rhsFactor));

      builder.create<EquationSidesOp>(loc, lhs, rhs);
      terminator->erase();

      return mlir::success();
    }

    if (lhsHasAccess) {
      auto leftPos = llvm::partition(lhsSummedValues, [&](const auto& value) {
        return containsAccessFn(value, access, EquationPath::LEFT);
      });

      mlir::Value lhsFactor = std::accumulate(
          lhsSummedValues.begin(), leftPos,
          builder.create<ConstantOp>(getOperation()->getLoc(), RealAttribute::get(builder.getContext(), 0)).getResult(),
          [&](mlir::Value acc, mlir::Value value) -> mlir::Value {
            mlir::Value factor = getMultiplyingFactor(builder, value, access.getVariable()->getValue(), access.getAccessFunction());
            return builder.create<AddOp>(value.getLoc(), getMostGenericType(acc.getType(), value.getType()), acc, factor);
          });

      mlir::Value lhsRemaining = std::accumulate(
          leftPos, lhsSummedValues.end(),
          builder.create<ConstantOp>(getOperation()->getLoc(), RealAttribute::get(builder.getContext(), 0)).getResult(),
          [&](mlir::Value acc, mlir::Value value) -> mlir::Value {
            return builder.create<AddOp>(value.getLoc(), getMostGenericType(acc.getType(), value.getType()), acc, value);
          });

      auto terminator = getTerminator();
      auto loc = terminator->getLoc();

      auto lhs = getValueAtPath(access.getPath());

      mlir::Value rhs = builder.create<DivOp>(
          loc, lhs.getType(),
          builder.create<SubOp>(loc, getMostGenericType(terminator.rhs()[0].getType(), lhsRemaining.getType()), terminator.rhs()[0], lhsRemaining),
          lhsFactor);

      builder.create<EquationSidesOp>(loc, lhs, rhs);
      terminator->erase();

      return mlir::success();
    }

    if (rhsHasAccess) {
      auto rightPos = llvm::partition(rhsSummedValues, [&](const auto& value) {
        return containsAccessFn(value, access, EquationPath::RIGHT);
      });

      mlir::Value rhsFactor = std::accumulate(
          rhsSummedValues.begin(), rightPos,
          builder.create<ConstantOp>(getOperation()->getLoc(), RealAttribute::get(builder.getContext(), 0)).getResult(),
          [&](mlir::Value acc, mlir::Value value) -> mlir::Value {
            mlir::Value factor = getMultiplyingFactor(builder, value, access.getVariable()->getValue(), access.getAccessFunction());
            return builder.create<AddOp>(value.getLoc(), getMostGenericType(acc.getType(), value.getType()), acc, factor);
          });

      mlir::Value rhsRemaining = std::accumulate(
          rightPos, rhsSummedValues.end(),
          builder.create<ConstantOp>(getOperation()->getLoc(), RealAttribute::get(builder.getContext(), 0)).getResult(),
          [&](mlir::Value acc, mlir::Value value) -> mlir::Value {
            return builder.create<AddOp>(value.getLoc(), getMostGenericType(acc.getType(), value.getType()), acc, value);
          });

      auto terminator = getTerminator();
      auto loc = terminator->getLoc();

      auto lhs = getValueAtPath(access.getPath());

      mlir::Value rhs = builder.create<DivOp>(
          loc, lhs.getType(),
          builder.create<SubOp>(loc, getMostGenericType(terminator.lhs()[0].getType(), rhsRemaining.getType()), terminator.lhs()[0], rhsRemaining),
          rhsFactor);

      builder.create<EquationSidesOp>(loc, lhs, rhs);
      terminator->erase();

      return mlir::success();
    }

    llvm_unreachable("Access not found");
    return mlir::failure();
  }

  std::pair<mlir::Value, std::vector<mlir::Value>> BaseEquation::collectSubscriptionIndexes(mlir::Value value) const
  {
    return ::collectSubscriptionIndexes(value);
  }

  mlir::Value BaseEquation::getMultiplyingFactor(
      mlir::OpBuilder& builder,
      mlir::Value value,
      mlir::Value variable,
      const AccessFunction& accessFunction) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto subscription = collectSubscriptionIndexes(value);

    if (subscription.first == variable) {
      std::vector<DimensionAccess> dimensionAccesses;

      for (mlir::Value index : subscription.second) {
        dimensionAccesses.push_back(resolveDimensionAccess(evaluateDimensionAccess(index)));
      }

      if (accessFunction == AccessFunction(std::move(dimensionAccesses))) {
        return builder.create<ConstantOp>(value.getLoc(), getIntegerAttribute(builder, value.getType(), 1));
      }
    }

    mlir::Operation* op = value.getDefiningOp();

    if (auto constantOp = mlir::dyn_cast<ConstantOp>(op)) {
      return constantOp.getResult();
    }

    if (auto negateOp = mlir::dyn_cast<NegateOp>(op)) {
      mlir::Value operand = getMultiplyingFactor(builder, negateOp.operand(), variable, accessFunction);
      return builder.create<NegateOp>(negateOp.getLoc(), negateOp.resultType(), operand);
    }

    if (auto mulOp = mlir::dyn_cast<MulOp>(op)) {
      mlir::Value lhs = getMultiplyingFactor(builder, mulOp.lhs(), variable, accessFunction);
      mlir::Value rhs = getMultiplyingFactor(builder, mulOp.rhs(), variable, accessFunction);
      return builder.create<MulOp>(mulOp.getLoc(), mulOp.resultType(), lhs, rhs);
    }

    if (auto divOp = mlir::dyn_cast<DivOp>(op)) {
      mlir::Value dividend = getMultiplyingFactor(builder, divOp.lhs(), variable, accessFunction);
      return builder.create<DivOp>(divOp.getLoc(), divOp.resultType(), dividend, divOp.rhs());
    }

    return value;
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
