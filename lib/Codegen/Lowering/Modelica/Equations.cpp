#include "marco/Codegen/Lowering/Modelica/LowererImpl.h"

using namespace ::marco;
using namespace ::marco::ast::modelica;
using namespace ::marco::codegen::lowering::modelica;
using namespace ::mlir::modelica;

namespace {
/// Create an operation without operands and results, with the given number
/// of empty regions.
template <typename Op>
Op createOp(mlir::OpBuilder &builder, mlir::Location location,
            unsigned numRegions = 0) {
  mlir::OperationState state(location, Op::getOperationName());

  for (unsigned i = 0; i < numRegions; ++i) {
    state.addRegion();
  }

  return mlir::cast<Op>(builder.create(state));
}

/// Built-in operators with function syntax that are used as equations or
/// statements.
enum class ProceduralBuiltin {
  None,
  Assert,
  Terminate,
  Reinit,
  ConnectionsBranch,
  ConnectionsRoot,
  ConnectionsPotentialRoot,
  ConnectionsUniqueRoot
};
} // namespace

namespace marco::codegen::lowering::modelica {
namespace {
/// Get the built-in operator called by a function call, if any. User
/// definitions with the same name take precedence.
ProceduralBuiltin getProceduralBuiltin(LowererImpl &lowerer, const Call &call) {
  Name name = call.getCallee()->getName();

  if (name.global) {
    return ProceduralBuiltin::None;
  }

  if (name.parts.size() == 1) {
    if (lowerer.classTree.lookup(lowerer.scope, name.parts[0])) {
      return ProceduralBuiltin::None;
    }

    return llvm::StringSwitch<ProceduralBuiltin>(name.parts[0])
        .Case("assert", ProceduralBuiltin::Assert)
        .Case("terminate", ProceduralBuiltin::Terminate)
        .Case("reinit", ProceduralBuiltin::Reinit)
        .Default(ProceduralBuiltin::None);
  }

  if (name.parts.size() == 2) {
    auto element = lowerer.classTree.lookup(lowerer.scope, name.parts[0]);

    if (!element || !element->isClass() ||
        element->classNode->getPredefined() != PredefinedClass::Connections) {
      return ProceduralBuiltin::None;
    }

    return llvm::StringSwitch<ProceduralBuiltin>(name.parts[1])
        .Case("branch", ProceduralBuiltin::ConnectionsBranch)
        .Case("root", ProceduralBuiltin::ConnectionsRoot)
        .Case("potentialRoot", ProceduralBuiltin::ConnectionsPotentialRoot)
        .Case("uniqueRoot", ProceduralBuiltin::ConnectionsUniqueRoot)
        .Default(ProceduralBuiltin::None);
  }

  return ProceduralBuiltin::None;
}

/// Lower a call to a built-in procedural operator into the operation
/// representing it.
template <typename Op>
mlir::LogicalResult lowerProcedural(LowererImpl &lowerer, const Call &call,
                                    llvm::ArrayRef<llvm::StringRef> parameters,
                                    size_t minArguments) {
  auto arguments =
      lowerer.getBuiltinArguments(call, parameters, minArguments);

  if (mlir::failed(arguments)) {
    return mlir::failure();
  }

  mlir::Location location = lowerer.loc(call);

  if (llvm::is_contained(*arguments, nullptr)) {
    return lowerer.emitError(location, "missing arguments");
  }

  auto op = createOp<Op>(lowerer.builder, location, 1);

  if (mlir::failed(
          lowerer.lowerIntoRegion(op.getBodyRegion(), *arguments, location))) {
    op.erase();
    return mlir::failure();
  }

  return mlir::success();
}

/// Lower a call to a procedural built-in operator, or to a user function
/// whose outputs are not used.
template <typename CallOp>
mlir::LogicalResult lowerProceduralCall(LowererImpl &lowerer,
                                        const Call &call) {
  switch (getProceduralBuiltin(lowerer, call)) {
  case ProceduralBuiltin::Assert:
    return lowerProcedural<AssertOp>(lowerer, call,
                                     {"condition", "message", "level"}, 2);
  case ProceduralBuiltin::Terminate:
    return lowerProcedural<TerminateOp>(lowerer, call, {"message"}, 1);
  case ProceduralBuiltin::Reinit:
    return lowerProcedural<ReinitOp>(lowerer, call, {"x", "expr"}, 2);
  case ProceduralBuiltin::ConnectionsBranch:
    return lowerProcedural<ConnectionsBranchOp>(lowerer, call, {"a", "b"}, 2);
  case ProceduralBuiltin::ConnectionsRoot:
    return lowerProcedural<ConnectionsRootOp>(lowerer, call, {"a"}, 1);
  case ProceduralBuiltin::ConnectionsPotentialRoot:
    return lowerProcedural<ConnectionsPotentialRootOp>(
        lowerer, call, {"a", "priority"}, 1);
  case ProceduralBuiltin::ConnectionsUniqueRoot:
    return lowerProcedural<ConnectionsUniqueRootOp>(lowerer, call,
                                                    {"a", "message"}, 2);
  case ProceduralBuiltin::None:
    break;
  }

  if (!call.getIterators().empty()) {
    return lowerer.emitError(lowerer.loc(call),
                             "iterators are not allowed in procedure calls");
  }

  mlir::Location location = lowerer.loc(call);
  auto op = createOp<CallOp>(lowerer.builder, location, 1);
  op.setCalleeAttr(lowerer.getSymbolRef(call.getCallee()->getName()));

  llvm::SmallVector<const Expression *> arguments;
  llvm::SmallVector<llvm::StringRef> argNames;

  for (const Child<Expression> &argument : call.getArguments()) {
    arguments.push_back(argument.get());
  }

  for (const Child<NamedArgument> &argument : call.getNamedArguments()) {
    arguments.push_back(argument->getValue());
    argNames.push_back(argument->getName());
  }

  if (!argNames.empty()) {
    op.setArgNamesAttr(lowerer.builder.getStrArrayAttr(argNames));
  }

  mlir::OpBuilder::InsertionGuard guard(lowerer.builder);

  if (mlir::failed(
          lowerer.lowerIntoRegion(op.getBodyRegion(), arguments, location))) {
    op.erase();
    return mlir::failure();
  }

  return mlir::success();
}

/// Lower the sides of an equation or assignment whose left-hand side may be
/// a list of outputs, as in `(a, , b) = f(x)`. The outputs omitted in the
/// list are omitted from the right-hand side values.
mlir::LogicalResult
lowerSides(LowererImpl &lowerer, llvm::ArrayRef<const Expression *> lhs,
           const Expression &rhs, mlir::Location location,
           llvm::SmallVectorImpl<mlir::Value> &lhsValues,
           llvm::SmallVectorImpl<mlir::Value> &rhsValues) {
  llvm::SmallVector<size_t> positions;

  for (auto [index, expression] : llvm::enumerate(lhs)) {
    if (!expression) {
      continue;
    }

    auto value = lowerer.lowerExpression(*expression);

    if (mlir::failed(value)) {
      return mlir::failure();
    }

    lhsValues.push_back(*value);
    positions.push_back(index);
  }

  if (lhs.size() == 1) {
    auto value = lowerer.lowerExpression(rhs);

    if (mlir::failed(value)) {
      return mlir::failure();
    }

    rhsValues.push_back(*value);
    return mlir::success();
  }

  auto results = lowerer.lowerExpressionResults(rhs);

  if (mlir::failed(results)) {
    return mlir::failure();
  }

  if (results->size() < lhs.size()) {
    return lowerer.emitError(location, "the right-hand side has " +
                                           llvm::Twine(results->size()) +
                                           " values, but " +
                                           llvm::Twine(lhs.size()) +
                                           " are expected");
  }

  for (size_t position : positions) {
    rhsValues.push_back((*results)[position]);
  }

  return mlir::success();
}

/// Get the left-hand side expressions of an equation.
llvm::SmallVector<const Expression *> getOutputs(const Expression &lhs) {
  llvm::SmallVector<const Expression *> result;

  if (auto outputList = lhs.dyn_cast<OutputList>()) {
    for (const Child<Expression> &element : outputList->getElements()) {
      result.push_back(element.get());
    }
  } else {
    result.push_back(&lhs);
  }

  return result;
}

/// Lower a for-loop with the given indices. The body is lowered by the
/// callback once all the loop variables are available.
template <typename Op>
mlir::LogicalResult
lowerForLoop(LowererImpl &lowerer, const Children<ForIndex> &indices,
             size_t current, mlir::Location location,
             llvm::function_ref<mlir::LogicalResult()> lowerBody) {
  if (current == indices.size()) {
    return lowerBody();
  }

  const ForIndex &index = *indices[current];
  mlir::OpBuilder &builder = lowerer.builder;
  auto op = createOp<Op>(builder, location, 2);

  mlir::Type inductionType = IntegerType::get(builder.getContext());

  if (const Expression *range = index.getRange()) {
    if (mlir::failed(
            lowerer.lowerIntoRegion(op.getRangeRegion(), {range}, location))) {
      op.erase();
      return mlir::failure();
    }

    auto yieldOp =
        mlir::cast<YieldOp>(op.getRangeRegion().front().getTerminator());

    mlir::Type rangeType = yieldOp.getValues()[0].getType();
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(rangeType);

    if (!tensorType || tensorType.getRank() != 1) {
      op.erase();
      return lowerer.emitError(lowerer.loc(index),
                               "the range of a loop must be a vector");
    }

    inductionType = tensorType.getElementType();
  }

  mlir::Block *body = &op.getBodyRegion().emplaceBlock();
  mlir::Value inductionVariable =
      body->addArgument(inductionType, lowerer.loc(index));

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  lowerer.pushIterator(index.getName(), inductionVariable);

  mlir::LogicalResult result =
      lowerForLoop<Op>(lowerer, indices, current + 1, location, lowerBody);

  lowerer.popIterator();

  if (mlir::failed(result)) {
    op.erase();
  }

  return result;
}

/// Lower a chain of conditional branches (if or when), nesting the
/// subsequent branches into the else region of the previous one.
template <typename Op, typename Body>
mlir::LogicalResult
lowerConditionalChain(LowererImpl &lowerer, const Children<Expression> &conditions,
                      const std::vector<Body> &branches, size_t current,
                      const Body *elseBranch, mlir::Location location,
                      llvm::function_ref<mlir::LogicalResult(const Body &)> lowerBody) {
  mlir::OpBuilder &builder = lowerer.builder;
  auto op = createOp<Op>(builder, location, 3);

  auto lower = [&]() -> mlir::LogicalResult {
    if (mlir::failed(lowerer.lowerIntoRegion(
            op->getRegion(0), {conditions[current].get()}, location))) {
      return mlir::failure();
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&op->getRegion(1).emplaceBlock());

    if (mlir::failed(lowerBody(branches[current]))) {
      return mlir::failure();
    }

    bool hasNextBranch = current + 1 < branches.size();

    if (!hasNextBranch && (!elseBranch || elseBranch->empty())) {
      return mlir::success();
    }

    builder.setInsertionPointToStart(&op->getRegion(2).emplaceBlock());

    if (hasNextBranch) {
      return lowerConditionalChain<Op, Body>(lowerer, conditions, branches,
                                             current + 1, elseBranch, location,
                                             lowerBody);
    }

    return lowerBody(*elseBranch);
  };

  if (mlir::failed(lower())) {
    op->erase();
    return mlir::failure();
  }

  return mlir::success();
}
} // namespace

//===---------------------------------------------------------------------===//
// Sections
//===---------------------------------------------------------------------===//

mlir::LogicalResult
LowererImpl::lowerEquationSection(const EquationSection &section) {
  auto op = createOp<EquationSectionOp>(builder, loc(section), 1);
  op.setIsInitial(section.isInitial());

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&op.getBodyRegion().emplaceBlock());
  bool failed = false;

  // The equations whose lowering fails are reported and omitted.
  for (const Child<Equation> &equation : section.getEquations()) {
    failed |= mlir::failed(lowerEquation(*equation));
  }

  return mlir::failure(failed);
}

mlir::LogicalResult
LowererImpl::lowerAlgorithmSection(const AlgorithmSection &section) {
  auto op = createOp<AlgorithmSectionOp>(builder, loc(section), 1);
  op.setIsInitial(section.isInitial());

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&op.getBodyRegion().emplaceBlock());
  bool failed = false;

  // The statements whose lowering fails are reported and omitted.
  for (const Child<Statement> &statement : section.getStatements()) {
    failed |= mlir::failed(lowerStatement(*statement));
  }

  return mlir::failure(failed);
}

//===---------------------------------------------------------------------===//
// Equations
//===---------------------------------------------------------------------===//

mlir::LogicalResult
LowererImpl::lowerEquations(const Children<Equation> &equations) {
  for (const Child<Equation> &equation : equations) {
    if (mlir::failed(lowerEquation(*equation))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult LowererImpl::lowerEquation(const Equation &equation) {
  mlir::Location location = loc(equation);

  if (auto equality = equation.dyn_cast<EqualityEquation>()) {
    auto op = createOp<EquationOp>(builder, location, 1);
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&op.getBodyRegion().emplaceBlock());

    llvm::SmallVector<mlir::Value> lhs;
    llvm::SmallVector<mlir::Value> rhs;

    if (mlir::failed(lowerSides(*this, getOutputs(*equality->getLhs()),
                                *equality->getRhs(), location, lhs, rhs))) {
      op.erase();
      return mlir::failure();
    }

    builder.create<EquationSidesOp>(location, lhs, rhs);
    return mlir::success();
  }

  if (auto callEquation = equation.dyn_cast<CallEquation>()) {
    return lowerProceduralCall<CallEquationOp>(*this, *callEquation->getCall());
  }

  if (auto connect = equation.dyn_cast<ConnectEquation>()) {
    auto op = createOp<ConnectOp>(builder, location, 1);

    if (mlir::failed(lowerIntoRegion(op.getBodyRegion(),
                                     {connect->getLhs(), connect->getRhs()},
                                     location))) {
      op.erase();
      return mlir::failure();
    }

    return mlir::success();
  }

  if (auto ifEquation = equation.dyn_cast<IfEquation>()) {
    return lowerConditionalChain<IfEquationOp, Children<Equation>>(
        *this, ifEquation->getConditions(), ifEquation->getBranches(), 0,
        &ifEquation->getElseEquations(), location,
        [&](const Children<Equation> &body) { return lowerEquations(body); });
  }

  if (auto whenEquation = equation.dyn_cast<WhenEquation>()) {
    return lowerConditionalChain<WhenEquationOp, Children<Equation>>(
        *this, whenEquation->getConditions(), whenEquation->getBranches(), 0,
        nullptr, location,
        [&](const Children<Equation> &body) { return lowerEquations(body); });
  }

  if (auto forEquation = equation.dyn_cast<ForEquation>()) {
    return lowerForLoop<ForEquationOp>(
        *this, forEquation->getIndices(), 0, location,
        [&]() { return lowerEquations(forEquation->getEquations()); });
  }

  return emitError(location, "unexpected equation");
}

//===---------------------------------------------------------------------===//
// Statements
//===---------------------------------------------------------------------===//

mlir::LogicalResult
LowererImpl::lowerStatements(const Children<Statement> &statements) {
  for (const Child<Statement> &statement : statements) {
    if (mlir::failed(lowerStatement(*statement))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult LowererImpl::lowerStatement(const Statement &statement) {
  mlir::Location location = loc(statement);

  auto lowerAssignment = [&](llvm::ArrayRef<const Expression *> destinations,
                             const Expression &value) -> mlir::LogicalResult {
    auto op = createOp<AssignmentOp>(builder, location, 1);
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&op.getBodyRegion().emplaceBlock());

    llvm::SmallVector<mlir::Value> lhs;
    llvm::SmallVector<mlir::Value> rhs;

    if (mlir::failed(lowerSides(*this, destinations, value, location, lhs,
                                rhs))) {
      op.erase();
      return mlir::failure();
    }

    builder.create<AssignmentSidesOp>(location, lhs, rhs);
    return mlir::success();
  };

  if (auto assignment = statement.dyn_cast<AssignmentStatement>()) {
    return lowerAssignment({assignment->getDestination()},
                           *assignment->getValue());
  }

  if (auto assignment = statement.dyn_cast<MultipleAssignmentStatement>()) {
    llvm::SmallVector<const Expression *> destinations;

    for (const Child<Expression> &destination : assignment->getDestinations()) {
      destinations.push_back(destination.get());
    }

    return lowerAssignment(destinations, *assignment->getCall());
  }

  if (auto callStatement = statement.dyn_cast<CallStatement>()) {
    return lowerProceduralCall<CallStatementOp>(*this,
                                                *callStatement->getCall());
  }

  if (statement.isa<BreakStatement>()) {
    builder.create<BreakOp>(location);
    return mlir::success();
  }

  if (statement.isa<ReturnStatement>()) {
    builder.create<ReturnOp>(location);
    return mlir::success();
  }

  if (auto ifStatement = statement.dyn_cast<IfStatement>()) {
    return lowerConditionalChain<IfOp, Children<Statement>>(
        *this, ifStatement->getConditions(), ifStatement->getBranches(), 0,
        &ifStatement->getElseStatements(), location,
        [&](const Children<Statement> &body) { return lowerStatements(body); });
  }

  if (auto whenStatement = statement.dyn_cast<WhenStatement>()) {
    return lowerConditionalChain<WhenOp, Children<Statement>>(
        *this, whenStatement->getConditions(), whenStatement->getBranches(), 0,
        nullptr, location,
        [&](const Children<Statement> &body) { return lowerStatements(body); });
  }

  if (auto forStatement = statement.dyn_cast<ForStatement>()) {
    return lowerForLoop<ForOp>(
        *this, forStatement->getIndices(), 0, location,
        [&]() { return lowerStatements(forStatement->getStatements()); });
  }

  if (auto whileStatement = statement.dyn_cast<WhileStatement>()) {
    auto op = createOp<WhileOp>(builder, location, 2);

    auto lower = [&]() -> mlir::LogicalResult {
      if (mlir::failed(lowerIntoRegion(op.getConditionRegion(),
                                       {whileStatement->getCondition()},
                                       location))) {
        return mlir::failure();
      }

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&op.getBodyRegion().emplaceBlock());
      return lowerStatements(whileStatement->getStatements());
    };

    if (mlir::failed(lower())) {
      op.erase();
      return mlir::failure();
    }

    return mlir::success();
  }

  return emitError(location, "unexpected statement");
}
} // namespace marco::codegen::lowering::modelica
