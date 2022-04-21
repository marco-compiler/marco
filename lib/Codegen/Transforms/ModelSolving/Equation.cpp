#include "marco/Codegen/Transforms/ModelSolving/Equation.h"
#include "marco/Codegen/Transforms/ModelSolving/EquationImpl.h"
#include "marco/Codegen/Transforms/ModelSolving/LoopEquation.h"
#include "marco/Codegen/Transforms/ModelSolving/ScalarEquation.h"
#include "marco/Codegen/Utils.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/ADT/STLExtras.h"
#include <numeric>

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

static long getIntFromAttribute(mlir::Attribute attribute)
{
  if (auto indexAttr = attribute.dyn_cast<mlir::IntegerAttr>()) {
    return indexAttr.getInt();
  }

  if (auto booleanAttr = attribute.dyn_cast<BooleanAttr>()) {
    return booleanAttr.getValue() ? 1 : 0;
  }

  if (auto integerAttr = attribute.dyn_cast<IntegerAttr>()) {
    return integerAttr.getValue().getSExtValue();
  }

  if (auto realAttr = attribute.dyn_cast<RealAttr>()) {
    return realAttr.getValue().convertToDouble();
  }

  llvm_unreachable("Unknown attribute type");
  return 0;
}

static mlir::Attribute getIntegerAttribute(mlir::OpBuilder& builder, mlir::Type type, int value)
{
  if (type.isa<BooleanType>()) {
    return BooleanAttr::get(type, value > 0);
  }

  if (type.isa<IntegerType>()) {
    return IntegerAttr::get(type, value);
  }

  if (type.isa<RealType>()) {
    return RealAttr::get(type, value);
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
    auto terminator = mlir::cast<EquationSidesOp>(equation.bodyBlock()->getTerminator());

    return llvm::any_of(terminator.lhsValues(), [](mlir::Value value) {
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

      value = loadOp.array();
      op = value.getDefiningOp();
    } else {
      auto subscriptionOp = mlir::cast<SubscriptionOp>(op);
      auto subscriptionIndexes = subscriptionOp.indices();

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
    auto addOp = builder.create<AddOp>(subOp->getLoc(), subOp.getResult().getType(), subOp.lhs(), negatedRhs);
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

  if (auto distributableOp = mlir::dyn_cast<DistributableOpInterface>(op)) {
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

namespace marco::codegen
{
  std::unique_ptr<Equation> Equation::build(mlir::modelica::EquationOp equation, Variables variables)
  {
    if (hasInductionVariables(equation)) {
      return std::make_unique<LoopEquation>(std::move(equation), std::move(variables));
    }

    return std::make_unique<ScalarEquation>(std::move(equation), std::move(variables));
  }

  Equation::~Equation() = default;

  void Equation::dumpIR() const
  {
    dumpIR(llvm::dbgs());
  }

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
      return isReferenceAccess(loadOp.array());
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
      searchAccesses(accesses, loadOp.array(), dimensionAccesses, std::move(path));
    } else if (auto subscriptionOp = mlir::dyn_cast<SubscriptionOp>(op)) {
      processIndexesFn(subscriptionOp.indices());
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
        } else {
          // If the variable is not subscribed enough times, then the remaining indices must be
          // added in their full ranges.

          std::vector<Range> additionalRanges;
          auto shape = arrayType.getShape();

          for (size_t i = shape.size() - dimensionsAccesses.size(); i < shape.size(); ++i) {
            additionalRanges.push_back(modeling::Range(0, shape[i]));
          }

          MultidimensionalRange additionalMultidimensionalRange(additionalRanges);

          for (const auto& indices : additionalMultidimensionalRange) {
            std::vector<DimensionAccess> completeDimensionsAccesses(reverted.begin(), reverted.end());

            for (const auto& index : indices) {
              completeDimensionsAccesses.push_back(DimensionAccess::constant(index));
            }

            accesses.emplace_back(*variable, AccessFunction(completeDimensionsAccesses), std::move(path));
          }
        }
      }
    }
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

  BaseEquation::BaseEquation(mlir::modelica::EquationOp equation, Variables variables)
      : equationOp(equation.getOperation()),
        variables(std::move(variables))
  {
    assert(getTerminator().lhsValues().size() == 1);
    assert(getTerminator().rhsValues().size() == 1);
  }

  mlir::modelica::EquationOp BaseEquation::getOperation() const
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
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().bodyBlock()->getTerminator());
    mlir::Value value = side == EquationPath::LEFT ? terminator.lhsValues()[0] : terminator.rhsValues()[0];

    for (auto index : path) {
      mlir::Operation* op = value.getDefiningOp();
      assert(op != nullptr && "Invalid equation path");
      value = op->getOperand(index);
    }

    return value;
  }

  mlir::LogicalResult BaseEquation::explicitate(
      mlir::OpBuilder& builder, const MultidimensionalRange& equationIndices, const EquationPath& path)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Get all the paths that lead to accesses with the same accessed variable
    // and function.
    auto requestedAccess = getAccessAtPath(path);
    std::vector<Access> accesses;

    for (const auto& access : getAccesses()) {
      if (requestedAccess.getVariable() != access.getVariable()) {
        continue;
      }

      auto requestedIndices = requestedAccess.getAccessFunction().map(equationIndices);
      auto currentIndices = access.getAccessFunction().map(equationIndices);

      assert(requestedIndices == currentIndices || !requestedIndices.overlaps(currentIndices));

      if (requestedIndices == currentIndices) {
        accesses.push_back(access);
      }
    }

    assert(!accesses.empty());

    // If there is only one access, then it is sufficient to follow the path
    // and invert the operations.

    if (accesses.size() == 1) {
      auto terminator = mlir::cast<EquationSidesOp>(getOperation().bodyBlock()->getTerminator());

      auto lhsOp = terminator.lhs().getDefiningOp();
      auto rhsOp = terminator.rhs().getDefiningOp();

      builder.setInsertionPoint(lhsOp);

      if (rhsOp->isBeforeInBlock(lhsOp)) {
        builder.setInsertionPoint(rhsOp);
      }

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

      if (auto res = groupLeftHandSide(builder, equationIndices, requestedAccess); mlir::failed(res)) {
        return res;
      }
    }

    return mlir::success();
  }

  std::unique_ptr<Equation> BaseEquation::cloneIRAndExplicitate(
      mlir::OpBuilder& builder, const MultidimensionalRange& equationIndices, const EquationPath& path) const
  {
    EquationOp clonedOp = cloneIR();
    auto result = Equation::build(clonedOp, getVariables());

    if (auto res = result->explicitate(builder, equationIndices, path); mlir::failed(res)) {
      result->eraseIR();
      return nullptr;
    }

    return result;
  }

  mlir::LogicalResult BaseEquation::replaceInto(
      mlir::OpBuilder& builder,
      const MultidimensionalRange& equationIndices,
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
    mlir::Operation* insertionPoint = destination.getOperation().bodyBlock()->getTerminator();

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
    auto sourceAccess = getAccessAtPath(EquationPath::LEFT);
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
      // For example, given the equation "x[10, i1] = ..." , with i1 belonging to [5,6),
      // then i1 can be replaced everywhere within the equation with the constant value
      // 5. Then, if we consider just the [i1] access of 'x', the reduced access
      // function can be now inverted and combined with the destination access, as
      // in the previous case.
      // Note that this always happens in case of scalar variables, as they are accessed
      // by means of a dummy access to their first element, as if they were arrays.

      llvm::SmallVector<bool, 3> usedInductions(getNumOfIterationVars(), false);
      llvm::SmallVector<DimensionAccess, 3> reducedSourceAccesses;
      llvm::SmallVector<DimensionAccess, 3> reducedDestinationAccesses;

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

          if (equationIndices[usage.index()].size() != 1) {
            getOperation().emitError("The write access is not invertible");
            return mlir::failure();
          }
        }
      }

      AccessFunction reducedSourceAccessFunction(reducedSourceAccesses);
      AccessFunction reducedDestinationAccessFunction(reducedDestinationAccesses);

      // Before inverting the reduced access function, we need to remap the induction variable indices.
      // Access functions like [i3 - 1][i2] are not in fact invertible, as it expects to operate on 4 induction
      // variables. We first convert it to [i0 - 1][i1], keeping track of the mappings, and then invert it.

      llvm::SmallVector<DimensionAccess, 3> remappedReducedSourceAccesses;
      std::set<size_t> remappedSourceInductions;
      llvm::SmallVector<size_t, 3> sourceDimensionMapping(reducedSourceAccesses.size(), 0);

      for (const auto& dimensionAccess : llvm::enumerate(reducedSourceAccesses)) {
        assert(!dimensionAccess.value().isConstantAccess());
        auto inductionIndex = dimensionAccess.value().getInductionVariableIndex();
        remappedSourceInductions.insert(inductionIndex);
        sourceDimensionMapping[dimensionAccess.index()] = inductionIndex;
        remappedReducedSourceAccesses.push_back(DimensionAccess::relative(dimensionAccess.index(), dimensionAccess.value().getOffset()));
      }

      // The reduced access function is now invertible.
      // Invert the function and combine it with the destination access.

      AccessFunction remappedReducedSourceAccessFunction(remappedReducedSourceAccesses);
      auto combinedReducedAccess = reducedDestinationAccessFunction.combine(remappedReducedSourceAccessFunction.inverse());

      // Then, revert the mappings done previously.
      llvm::SmallVector<DimensionAccess, 3> transformationAccesses;
      size_t usedInductionIndex = 0;

      for (size_t i = 0, e = getNumOfIterationVars(); i < e; ++i) {
        if (usedInductions[i]) {
          transformationAccesses.push_back(combinedReducedAccess[usedInductionIndex]);

        } else {
          const auto& range = equationIndices[i];
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
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().bodyBlock()->getTerminator());
    mlir::Value replacement = terminator.rhsValues()[0];

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
    // This is required when the source equation has implicit loops.
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
    return mlir::cast<EquationSidesOp>(getOperation().bodyBlock()->getTerminator());
  }

  // TODO
  std::vector<Access> BaseEquation::getUniqueAccesses(std::vector<Access> accesses) const
  {
    std::vector<Access> result;
    std::set<std::pair<const Variable*, const AccessFunction*>> uniqueAccesses;

    /*
    for (const auto& access : accesses) {
      auto it = llvm::find_if(uniqueAccesses, [&](const auto& uniqueAccess) {
        return uniqueAccess->first == access.getVariable() && *uniqueAccess->second == access.getAccessFunction();
      });

      if (it != uniqueAccesses.end()) {
        result.push_back(access);
        uniqueAccesses.insert(std::make_pair<const Variable*, const AccessFunction*>(access.getVariable(), &access.getAccessFunction()));
      }
    }
     */

    return result;
  }

  mlir::LogicalResult BaseEquation::explicitate(
      mlir::OpBuilder& builder, size_t argumentIndex, EquationPath::EquationSide side)
  {
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().bodyBlock()->getTerminator());
    assert(terminator.lhsValues().size() == 1);
    assert(terminator.rhsValues().size() == 1);

    mlir::Value toExplicitate = side == EquationPath::LEFT ? terminator.lhsValues()[0] : terminator.rhsValues()[0];
    mlir::Value otherExp = side == EquationPath::RIGHT ? terminator.lhsValues()[0] : terminator.rhsValues()[0];

    mlir::Operation* op = toExplicitate.getDefiningOp();

    if (!op->hasTrait<InvertibleOpInterface::Trait>()) {
      op->emitError("Operation is not invertible");
      return mlir::failure();
    }

    return mlir::cast<InvertibleOpInterface>(op).invert(builder, argumentIndex, otherExp);
  }

  mlir::LogicalResult BaseEquation::groupLeftHandSide(
      mlir::OpBuilder& builder,
      const ::marco::modeling::MultidimensionalRange& equationIndices,
      const Access& access)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto lhs = getValueAtPath(access.getPath());

    // Determine whether the access to be grouped is inside both the equation's sides or just one of them.
    // When the requested access is found, also check that the path goes through linear operations. If not,
    // explicitation is not possible.
    bool lhsHasAccess = false;
    bool rhsHasAccess = false;

    for (const auto& acc : getAccesses()) {
      if (acc.getVariable() != access.getVariable()) {
        continue;
      }

      auto requestedIndices = access.getAccessFunction().map(equationIndices);
      auto currentIndices = acc.getAccessFunction().map(equationIndices);

      assert(requestedIndices == currentIndices || !requestedIndices.overlaps(currentIndices));

      if (requestedIndices == currentIndices) {
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
        return getTerminator().lhsValues()[0];
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
        return getTerminator().rhsValues()[0];
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
        if (acc.getVariable() != access.getVariable()) {
          return false;
        }

        auto requestedIndices = access.getAccessFunction().map(equationIndices);
        auto currentIndices = acc.getAccessFunction().map(equationIndices);

        assert(requestedIndices == currentIndices || !requestedIndices.overlaps(currentIndices));
        return requestedIndices == currentIndices;
      });
    };

    auto groupFactorsFn = [&](auto beginIt, auto endIt) -> mlir::Value {
      return std::accumulate(
          beginIt, endIt,
          builder.create<ConstantOp>(getOperation()->getLoc(), RealAttr::get(builder.getContext(), 0)).getResult(),
          [&](mlir::Value acc, mlir::Value value) -> mlir::Value {
            if (!acc) {
              return nullptr;
            }

            auto factor = getMultiplyingFactor(
                builder, equationIndices, value,
                access.getVariable()->getValue(),
                IndexSet(access.getAccessFunction().map(equationIndices)));

            if (!factor.second || factor.first > 1) {
              return nullptr;
            }

            return builder.create<AddOp>(
                value.getLoc(), getMostGenericType(acc.getType(), value.getType()),
                acc, factor.second);
          });
    };

    auto groupRemainingFn = [&](auto beginIt, auto endIt) -> mlir::Value {
      return std::accumulate(
          beginIt, endIt,
          builder.create<ConstantOp>(getOperation()->getLoc(), RealAttr::get(builder.getContext(), 0)).getResult(),
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

      mlir::Value rhs = builder.create<DivOp>(
          loc, lhs.getType(),
          builder.create<SubOp>(loc, getMostGenericType(rhsRemaining.getType(), lhsRemaining.getType()), rhsRemaining, lhsRemaining),
          builder.create<SubOp>(loc, getMostGenericType(lhsFactor.getType(), rhsFactor.getType()), lhsFactor, rhsFactor));

      auto lhsOp = builder.create<EquationSideOp>(loc, lhs);
      auto oldLhsOp = terminator.lhs().getDefiningOp();
      oldLhsOp->replaceAllUsesWith(lhsOp);
      oldLhsOp->erase();

      auto rhsOp = builder.create<EquationSideOp>(loc, rhs);
      auto oldRhsOp = terminator.rhs().getDefiningOp();
      oldRhsOp->replaceAllUsesWith(rhsOp);
      oldRhsOp->erase();

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

      mlir::Value rhs = builder.create<DivOp>(
          loc, lhs.getType(),
          builder.create<SubOp>(loc, getMostGenericType(terminator.rhsValues()[0].getType(), lhsRemaining.getType()), terminator.rhsValues()[0], lhsRemaining),
          lhsFactor);

      auto lhsOp = builder.create<EquationSideOp>(loc, lhs);
      auto oldLhsOp = terminator.lhs().getDefiningOp();
      oldLhsOp->replaceAllUsesWith(lhsOp);
      oldLhsOp->erase();

      auto rhsOp = builder.create<EquationSideOp>(loc, rhs);
      auto oldRhsOp = terminator.rhs().getDefiningOp();
      oldRhsOp->replaceAllUsesWith(rhsOp);
      oldRhsOp->erase();

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

      mlir::Value rhs = builder.create<DivOp>(
          loc, lhs.getType(),
          builder.create<SubOp>(loc, getMostGenericType(terminator.lhsValues()[0].getType(), rhsRemaining.getType()), terminator.lhsValues()[0], rhsRemaining),
          rhsFactor);

      auto lhsOp = builder.create<EquationSideOp>(loc, lhs);
      auto oldLhsOp = terminator.lhs().getDefiningOp();
      oldLhsOp->replaceAllUsesWith(lhsOp);
      oldLhsOp->erase();

      auto rhsOp = builder.create<EquationSideOp>(loc, rhs);
      auto oldRhsOp = terminator.rhs().getDefiningOp();
      oldRhsOp->replaceAllUsesWith(rhsOp);
      oldRhsOp->erase();

      return mlir::success();
    }

    llvm_unreachable("Access not found");
    return mlir::failure();
  }

  std::pair<unsigned int, mlir::Value> BaseEquation::getMultiplyingFactor(
      mlir::OpBuilder& builder,
      const MultidimensionalRange& equationIndices,
      mlir::Value value,
      mlir::Value variable,
      const IndexSet& variableIndices) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (isReferenceAccess(value)) {
      std::vector<Access> accesses;
      EquationPath path(EquationPath::LEFT);
      searchAccesses(accesses, value, path);
      assert(accesses.size() == 1);

      if (accesses[0].getVariable()->getValue() == variable &&
          variableIndices == accesses[0].getAccessFunction().map(equationIndices)) {
        mlir::Value one = builder.create<ConstantOp>(value.getLoc(), getIntegerAttribute(builder, value.getType(), 1));
        return std::make_pair(1, one);
      }
    }

    mlir::Operation* op = value.getDefiningOp();

    if (auto constantOp = mlir::dyn_cast<ConstantOp>(op)) {
      return std::make_pair(0, constantOp.getResult());
    }

    if (auto negateOp = mlir::dyn_cast<NegateOp>(op)) {
      auto operand = getMultiplyingFactor(builder, equationIndices, negateOp.operand(), variable, variableIndices);

      if (!operand.second) {
        return std::make_pair(operand.first, nullptr);
      }

      mlir::Value result = builder.create<NegateOp>(
          negateOp.getLoc(), negateOp.getResult().getType(), operand.second);

      return std::make_pair(operand.first, result);
    }

    if (auto mulOp = mlir::dyn_cast<MulOp>(op)) {
      auto lhs = getMultiplyingFactor(builder, equationIndices, mulOp.lhs(), variable, variableIndices);
      auto rhs = getMultiplyingFactor(builder, equationIndices, mulOp.rhs(), variable, variableIndices);

      if (!lhs.second || !rhs.second) {
        return std::make_pair(0, nullptr);
      }

      mlir::Value result = builder.create<MulOp>(
          mulOp.getLoc(), mulOp.getResult().getType(), lhs.second, rhs.second);

      return std::make_pair(lhs.first + rhs.first, result);
    }

    auto hasAccessToVar = [&](mlir::Value value) -> bool {
      // Dummy path. Not used, but required by the infrastructure.
      EquationPath path(EquationPath::LEFT);

      std::vector<Access> accesses;
      searchAccesses(accesses, value, path);

      bool hasAccess = llvm::any_of(accesses, [&](const auto& access) {
        return access.getVariable()->getValue() == variable &&
            variableIndices == access.getAccessFunction().map(equationIndices);
      });

      if (hasAccess) {
        return true;
      }

      return false;
    };

    if (auto divOp = mlir::dyn_cast<DivOp>(op)) {
      auto dividend = getMultiplyingFactor(builder, equationIndices, divOp.lhs(), variable, variableIndices);

      if (!dividend.second) {
        return dividend;
      }

      // Check that the right-hand side value has no access to the variable of interest
      if (hasAccessToVar(divOp.rhs())) {
        return std::make_pair(dividend.first, nullptr);
      }

      mlir::Value result = builder.create<DivOp>(
          divOp.getLoc(), divOp.getResult().getType(), dividend.second, divOp.rhs());

      return std::make_pair(dividend.first, result);
    }

    // Check that the value is not the result of an operation using the variable of interest.
    // If it has such access, then we are not able to extract the multiplying factor.
    if (hasAccessToVar(value)) {
      return std::make_pair(1, nullptr);
    }

    return std::make_pair(0, value);
  }

  void BaseEquation::createIterationLoops(
      mlir::OpBuilder& builder,
      mlir::Location loc,
      mlir::ValueRange beginIndices,
      mlir::ValueRange endIndices,
      mlir::ValueRange steps,
      marco::modeling::scheduling::Direction iterationDirection,
      std::function<void(mlir::OpBuilder&, mlir::ValueRange)> bodyBuilder) const
  {
    std::vector<mlir::Value> inductionVariables;

    assert(beginIndices.size() == endIndices.size());
    assert(beginIndices.size() == steps.size());

    assert(iterationDirection == modeling::scheduling::Direction::Forward ||
           iterationDirection == modeling::scheduling::Direction::Backward);

    auto conditionFn = [&](mlir::Value index, mlir::Value end) -> mlir::Value {
      if (iterationDirection == modeling::scheduling::Direction::Backward) {
        return builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sgt, index, end).getResult();
      }

      return builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::slt, index, end).getResult();
    };

    auto updateFn = [&](mlir::Value index, mlir::Value step) -> mlir::Value {
      if (iterationDirection == modeling::scheduling::Direction::Backward) {
        return builder.create<mlir::SubIOp>(loc, index, step).getResult();
      }

      return builder.create<mlir::AddIOp>(loc, index, step).getResult();
    };

    mlir::Operation* firstLoop = nullptr;

    for (size_t i = 0; i < steps.size(); ++i) {
      auto whileOp = builder.create<mlir::scf::WhileOp>(loc, builder.getIndexType(), beginIndices[i]);

      if (i == 0) {
        firstLoop = whileOp.getOperation();
      }

      // Check the condition.
      // A naive check can consist in the equality comparison. However, in order to be future-proof with
      // respect to steps greater than one, we need to check if the current value is beyond the end boundary.
      // This in turn requires to know the iteration direction.
      mlir::Block* beforeBlock = builder.createBlock(&whileOp.before(), {}, builder.getIndexType());
      builder.setInsertionPointToStart(beforeBlock);
      mlir::Value condition = conditionFn(whileOp.before().getArgument(0), endIndices[i]);
      builder.create<mlir::scf::ConditionOp>(loc, condition, whileOp.before().getArgument(0));

      // Execute the loop body
      mlir::Block* afterBlock = builder.createBlock(&whileOp.after(), {}, builder.getIndexType());
      mlir::Value inductionVariable = afterBlock->getArgument(0);
      inductionVariables.push_back(inductionVariable);
      builder.setInsertionPointToStart(afterBlock);

      // Update the induction variable
      mlir::Value nextValue = updateFn(inductionVariable, steps[i]);
      builder.create<mlir::scf::YieldOp>(loc, nextValue);
      builder.setInsertionPoint(nextValue.getDefiningOp());
    }

    bodyBuilder(builder, inductionVariables);

    if (firstLoop != nullptr) {
      builder.setInsertionPointAfter(firstLoop);
    }
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

    builder.setInsertionPointToEnd(&function.body().back());
    builder.create<mlir::ReturnOp>(loc);
    return function;
  }
}
