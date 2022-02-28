#include "llvm/ADT/STLExtras.h"
#include "marco/Codegen/Transforms/Model/Equation.h"
#include "marco/Codegen/Transforms/Model/EquationImpl.h"
#include "marco/Codegen/Transforms/Model/LoopEquation.h"
#include "marco/Codegen/Transforms/Model/ScalarEquation.h"
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

static mlir::LogicalResult removeSubtractions(mlir::OpBuilder& builder, mlir::Operation* root)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Operation* op = root;

  if (op == nullptr) {
    return mlir::success();
  }

  for (auto operand : op->getOperands()) {
    if (auto res = removeSubtractions(builder, operand.getDefiningOp()); mlir::failed(res)) {
      return res;
    }
  }

  if (auto subOp = mlir::dyn_cast<SubOp>(op)) {
    builder.setInsertionPoint(subOp);
    mlir::Value rhs = subOp.rhs();
    mlir::Value negatedRhs = builder.create<NegateOp>(rhs.getLoc(), rhs.getType(), rhs);
    auto addOp = builder.create<AddOp>(subOp->getLoc(), subOp.resultType(), subOp.lhs(), negatedRhs);
    subOp->replaceAllUsesWith(addOp.getOperation());
    subOp->erase();
  }

  return mlir::success();
}

static mlir::LogicalResult distributeMulAndDivOps(mlir::OpBuilder& builder, mlir::Operation* root)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Operation* op = root;

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
      builder.setInsertionPoint(distributableOp);
      mlir::Operation* result = distributableOp.distribute(builder).getDefiningOp();

      if (result != op) {
        op->replaceAllUsesWith(result);
        op->erase();
      }
    }
  }

  return mlir::success();
}

static mlir::LogicalResult pushNegateOps(mlir::OpBuilder& builder, mlir::Operation* root)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Operation* op = root;

  if (op == nullptr) {
    return mlir::success();
  }

  for (auto operand : op->getOperands()) {
    if (auto res = pushNegateOps(builder, operand.getDefiningOp()); mlir::failed(res)) {
      return res;
    }
  }

  if (auto distributableOp = mlir::dyn_cast<NegateOp>(op)) {
    builder.setInsertionPoint(distributableOp);
    mlir::Operation* result = distributableOp.distribute(builder).getDefiningOp();

    if (result != op) {
      op->replaceAllUsesWith(result);
      op->erase();
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

  Equation::~Equation() = default;

  llvm::Optional<Variable*> Equation::findVariable(mlir::Value value) const
  {
    assert(value.isa<mlir::BlockArgument>());
    auto variables = getVariables();

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
    if (isVariable(value)) {
      return true;
    }

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
        // Scalar variables are masked as arrays with just one element.
        // Thus, an access to a scalar variable is masked as an access to that unique element.

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

  TemporaryEquationGuard::TemporaryEquationGuard(Equation& equation) : equation(&equation)
  {
  }

  TemporaryEquationGuard::~TemporaryEquationGuard()
  {
    equation->eraseIR();
  }

  BaseEquation::BaseEquation(modelica::EquationOp equation, Variables variables)
      : equationOp(equation.getOperation()),
        variables(std::move(variables))
  {
    assert(getTerminator().lhs().size() == 1);
    assert(getTerminator().rhs().size() == 1);
  }

  modelica::EquationOp BaseEquation::getOperation() const
  {
    return mlir::cast<EquationOp>(equationOp);
  }

  Variables BaseEquation::getVariables() const
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
    } else {
      // If there are multiple accesses, then we must group all of them and
      // extract the common multiplying factor.

      if (auto res = groupLeftHandSide(builder, requestedAccess); mlir::failed(res)) {
        return res;
      }
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

    return result;
  }

  mlir::LogicalResult BaseEquation::replaceInto(
      mlir::OpBuilder& builder,
      Equation& destination,
      const ::marco::modeling::AccessFunction& destinationAccessFunction,
      const EquationPath& destinationPath) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Value valueToBeReplaced = destination.getValueAtPath(destinationPath);

    if (valueToBeReplaced.getUsers().empty()) {
      // Substitution is useless.
      // Just a safety check, normally not happening.
      return mlir::failure();
    }

    // Determine where the cloned operations will be placed, that is the first point
    // within the IR where the value to be replaced is used.
    mlir::Operation* insertionPoint = destination.getOperation().body()->getTerminator();

    for (const auto& user : valueToBeReplaced.getUsers()) {
      if (user->isBeforeInBlock(insertionPoint)) {
        insertionPoint = user;
      }
    }

    builder.setInsertionPoint(insertionPoint);

    // Map of the source equation values to the destination ones
    mlir::BlockAndValueMapping mapping;

    // Determine the access transformation to be applied to each induction variable usage.
    // For example, given the following equations:
    //   destination: x[i0, i1] = 1 - y[i1 + 3, i0 - 2]
    //   source:      y[i1 + 5, i0 - 1] = 3 - x[i0 + 1, i1 + 2] + z[i1 + 3, i0] + i1
    // In order to correctly insert the x[i0 + 1, i1 + 2] source access (and other ones,
    // if existing) into the destination  equation, some operations have to be performed.
    // First, the write access of the access must be inverted:
    //   ([i1 + 5, i0 - 1]) ^ (-1) = [i1 + 1, i0 - 5]
    // The destination access function is then composed with such inverted access:
    //   [i1 + 3, i0 - 2] * [i1 + 1, i0 - 5] = [i0 - 1, i1 - 2]
    // And finally it is combined with the access to be moved into the destination:
    //   [i0 - 1, i1 - 2] * [i0 + 1, i1 + 2] = [i0, i1]
    // In the same way, z[i1, i0] becomes z[i1 + 1, i0 - 1] and i1 becomes [i1 - 2].
    auto sourceAccess = getAccessFromPath(EquationPath::LEFT);
    const auto& sourceAccessFunction = sourceAccess.getAccessFunction();

    if (sourceAccessFunction.isInvertible()) {
      auto combinedAccess = destinationAccessFunction.combine(sourceAccess.getAccessFunction().inverse());

      if (auto res = mapInductionVariables(builder, mapping, destination, combinedAccess); mlir::failed(res)) {
        return res;
      }
    } else {
      // If the access function is not invertible, it may still be possible to move the
      // equation body. In fact, if all the induction variables not appearing in the
      // write access do iterate on a single value (i.e. [n,n+1)), then those constants
      // ('n', in the previous example), can be used to replace the induction variables
      // usages.
      // For example, given the equation "x[10, i1] = ..." , with i0 belonging to [5,6),
      // then i0 can be replaced everywhere within the equation with the constant value
      // 5. Then, if we consider just the [i1] access of 'x', the reduced access
      // function can be now inverted and combined with the destination access, as
      // in the previous case.
      // Note that this always happens in case of scalar variables, as they are accessed
      // by means of a fake access to their first element, as if they were arrays.

      llvm::SmallVector<bool, 3> usedInductions(sourceAccessFunction.size(), false);
      llvm::SmallVector<DimensionAccess, 3> reducedSourceAccesses;
      llvm::SmallVector<DimensionAccess, 3> reducedDestinationAccesses;
      auto iterationRanges = getIterationRanges();

      for (size_t i = 0, e = sourceAccessFunction.size(); i < e; ++i) {
        if (!sourceAccessFunction[i].isConstantAccess()) {
          usedInductions[sourceAccessFunction[i].getInductionVariableIndex()] = true;
          reducedSourceAccesses.push_back(sourceAccessFunction[i]);
          reducedDestinationAccesses.push_back(destinationAccessFunction[i]);
        }
      }

      for (const auto& usage : llvm::enumerate(usedInductions)) {
        if (!usage.value()) {
          // If the induction variable is not used, then ensure that it iterates
          // on just one value and thus can be replaced with a constant value.

          if (iterationRanges[usage.index()].size() != 1) {
            getOperation().emitError("The write access is not invertible");
            return mlir::failure();
          }
        }
      }

      AccessFunction reducedSourceAccessFunction(reducedSourceAccesses);
      AccessFunction reducedDestinationAccessFunction(reducedDestinationAccesses);

      llvm::SmallVector<DimensionAccess, 3> remappedReducedSourceAccesses;
      std::set<size_t> remappedSourceInductions;
      llvm::SmallVector<size_t, 3> sourceDimensionMapping(sourceAccessFunction.size(), 0);
      size_t mappedIndex = 0;

      for (const auto& dimensionAccess : reducedSourceAccesses) {
        assert(!dimensionAccess.isConstantAccess());
        auto inductionIndex = dimensionAccess.getInductionVariableIndex();
        remappedSourceInductions.insert(inductionIndex);
        sourceDimensionMapping[inductionIndex] = mappedIndex;
        remappedReducedSourceAccesses.push_back(DimensionAccess::relative(mappedIndex, dimensionAccess.getOffset()));
      }

      AccessFunction remappedReducedSourceAccessFunction(remappedReducedSourceAccesses);
      auto combinedReducedAccess = reducedDestinationAccessFunction.combine(remappedReducedSourceAccessFunction.inverse());
      llvm::SmallVector<DimensionAccess, 3> transformationAccesses;

      for (const auto& usage : llvm::enumerate(usedInductions)) {
        if (usage.value()) {
          assert(remappedSourceInductions.find(usage.index()) != remappedSourceInductions.end());
          transformationAccesses.push_back(combinedReducedAccess[sourceDimensionMapping[usage.index()]]);
        } else {
          const auto& range = iterationRanges[usage.index()];
          assert(range.size() == 1);
          transformationAccesses.push_back(DimensionAccess::constant(range.getBegin()));
        }
      }

      AccessFunction transformation(transformationAccesses);

      if (auto res = mapInductionVariables(builder, mapping, destination, transformation); mlir::failed(res)) {
        return res;
      }
    }

    // Obtain the value to be used for the replacement
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());
    mlir::Value replacement = terminator.rhs()[0];

    // The operations to be cloned, in reverse order
    std::vector<mlir::Operation*> toBeCloned;

    // Perform a depth-first traversal of the tree to determine which operations must
    // be cloned and in which order.
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
    mlir::Value mappedReplacement = mapping.lookup(replacement);

    // Add the missing subscriptions, if any.
    // This is required when the source equations has implicit loops.
    if (auto mappedReplacementArrayType = mappedReplacement.getType().dyn_cast<ArrayType>()) {
      size_t expectedRank = 0;

      if (auto originalArrayType = valueToBeReplaced.getType().dyn_cast<ArrayType>()) {
        expectedRank = originalArrayType.getRank();
      }

      if (mappedReplacementArrayType.getRank() > expectedRank) {
        auto originalIndexes = collectSubscriptionIndexes(valueToBeReplaced);
        size_t rankDifference = mappedReplacementArrayType.getRank() - expectedRank;
        std::vector<mlir::Value> additionalIndexes;

        for (size_t i = originalIndexes.second.size() - rankDifference; i < originalIndexes.second.size(); ++i) {
          additionalIndexes.push_back(originalIndexes.second[i]);
        }

        mlir::Value subscription = builder.create<SubscriptionOp>(
            mappedReplacement.getLoc(), mappedReplacement, additionalIndexes);

        mappedReplacement = builder.create<LoadOp>(mappedReplacement.getLoc(), subscription);
        mapping.map(replacement, mappedReplacement);
      }
    }

    valueToBeReplaced.replaceAllUsesWith(mappedReplacement);

    // Erase the replaced operations, which are now useless
    std::stack<mlir::Operation*> eraseStack;

    if (auto op = valueToBeReplaced.getDefiningOp(); op != nullptr) {
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

  // TODO
  std::vector<Access> BaseEquation::getUniqueAccesses(std::vector<Access> accesses) const
  {
    std::vector<Access> result;

    for (const auto& access : accesses) {

    }

    return result;
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
      op->emitError("Operation is not invertible");
      return mlir::failure();
    }

    return mlir::cast<InvertibleOpInterface>(op).invert(builder, argumentIndex, otherExp);
  }

  mlir::LogicalResult BaseEquation::groupLeftHandSide(mlir::OpBuilder& builder, const Access& access)
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
      }
    }

    // Convert the expression to a sum of values.
    auto convertToSumsFn = [&](std::function<mlir::Value()> root) -> mlir::LogicalResult {
      if (auto res = removeSubtractions(builder, root().getDefiningOp()); mlir::failed(res)) {
        return res;
      }

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

    auto groupFactorsFn = [&](auto beginIt, auto endIt) -> mlir::Value {
      return std::accumulate(
          beginIt, endIt,
          builder.create<ConstantOp>(getOperation()->getLoc(), RealAttribute::get(builder.getContext(), 0)).getResult(),
          [&](mlir::Value acc, mlir::Value value) -> mlir::Value {
            mlir::Value factor = getMultiplyingFactor(builder, value, access.getVariable()->getValue(), access.getAccessFunction());

            if (factor == nullptr) {
              return nullptr;
            }

            return builder.create<AddOp>(value.getLoc(), getMostGenericType(acc.getType(), value.getType()), acc, factor);
          });
    };

    auto groupRemainingFn = [&](auto beginIt, auto endIt) -> mlir::Value {
      return std::accumulate(
          beginIt, endIt,
          builder.create<ConstantOp>(getOperation()->getLoc(), RealAttribute::get(builder.getContext(), 0)).getResult(),
          [&](mlir::Value acc, mlir::Value value) -> mlir::Value {
            return builder.create<AddOp>(value.getLoc(), getMostGenericType(acc.getType(), value.getType()), acc, value);
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

      mlir::Value lhsFactor = groupFactorsFn(lhsSummedValues.begin(), leftPos);
      mlir::Value rhsFactor = groupFactorsFn(rhsSummedValues.begin(), rightPos);

      if (lhsFactor == nullptr || rhsFactor == nullptr) {
        return mlir::failure();
      }

      mlir::Value lhsRemaining = groupRemainingFn(leftPos, lhsSummedValues.end());
      mlir::Value rhsRemaining = groupRemainingFn(rightPos, rhsSummedValues.end());

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

      mlir::Value lhsFactor = groupFactorsFn(lhsSummedValues.begin(), leftPos);

      if (lhsFactor == nullptr) {
        return mlir::failure();
      }

      mlir::Value lhsRemaining = groupRemainingFn(leftPos, lhsSummedValues.end());

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

      mlir::Value rhsFactor = groupFactorsFn(rhsSummedValues.begin(), rightPos);

      if (rhsFactor == nullptr) {
        return mlir::failure();
      }

      mlir::Value rhsRemaining = groupRemainingFn(rightPos, rhsSummedValues.end());

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

  mlir::Value BaseEquation::getMultiplyingFactor(
      mlir::OpBuilder& builder,
      mlir::Value value,
      mlir::Value variable,
      const AccessFunction& accessFunction) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (isReferenceAccess(value)) {
      std::vector<Access> accesses;
      EquationPath path(EquationPath::LEFT);
      searchAccesses(accesses, value, path);
      assert(accesses.size() == 1);

      if (accesses[0].getVariable()->getValue() == variable && accesses[0].getAccessFunction() == accessFunction) {
        return builder.create<ConstantOp>(value.getLoc(), getIntegerAttribute(builder, value.getType(), 1));
      }
    }

    mlir::Operation* op = value.getDefiningOp();

    if (auto constantOp = mlir::dyn_cast<ConstantOp>(op)) {
      return constantOp.getResult();
    }

    if (auto negateOp = mlir::dyn_cast<NegateOp>(op)) {
      mlir::Value operand = getMultiplyingFactor(builder, negateOp.operand(), variable, accessFunction);

      if (operand == nullptr) {
        return nullptr;
      }

      return builder.create<NegateOp>(negateOp.getLoc(), negateOp.resultType(), operand);
    }

    if (auto mulOp = mlir::dyn_cast<MulOp>(op)) {
      mlir::Value lhs = getMultiplyingFactor(builder, mulOp.lhs(), variable, accessFunction);
      mlir::Value rhs = getMultiplyingFactor(builder, mulOp.rhs(), variable, accessFunction);

      if (lhs == nullptr || rhs == nullptr) {
        return nullptr;
      }

      return builder.create<MulOp>(mulOp.getLoc(), mulOp.resultType(), lhs, rhs);
    }

    auto hasAccessToVar = [&](mlir::Value value) -> bool {
      // Dummy path. Not used, but required by the infrastructure.
      EquationPath path(EquationPath::LEFT);

      std::vector<Access> accesses;
      searchAccesses(accesses, value, path);

      bool hasAccess = llvm::any_of(accesses, [&](const auto& access) {
        return access.getVariable()->getValue() == variable && access.getAccessFunction() == accessFunction;
      });

      if (hasAccess) {
        return true;
      }

      return false;
    };

    if (auto divOp = mlir::dyn_cast<DivOp>(op)) {
      mlir::Value dividend = getMultiplyingFactor(builder, divOp.lhs(), variable, accessFunction);

      if (dividend == nullptr) {
        return nullptr;
      }

      // Check that the right-hand side value has no access to the variable of interest
      if (hasAccessToVar(divOp.rhs())) {
        return nullptr;
      }

      return builder.create<DivOp>(divOp.getLoc(), divOp.resultType(), dividend, divOp.rhs());
    }

    // Check that the value is not the result of an operation using the variable of interest.
    // If it has such access, then we are not able to extract the multiplying factor.
    if (hasAccessToVar(value)) {
      return nullptr;
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
