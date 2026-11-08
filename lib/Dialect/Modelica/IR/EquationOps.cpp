#include "marco/Dialect/Modelica/IR/OpUtils.h"
#include "marco/Dialect/Modelica/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace ::mlir::modelica;
using namespace ::mlir::modelica::detail;

//===---------------------------------------------------------------------===//
// Utilities
//===---------------------------------------------------------------------===//

namespace {
/// Verify the values yielded by the region of a function call equation or
/// statement.
mlir::LogicalResult verifyCallArguments(mlir::Operation *op,
                                        mlir::Region &region,
                                        mlir::ArrayAttr argNames) {
  auto args = getYieldedValues(op, region, "body", 0,
                               std::numeric_limits<size_t>::max());

  if (mlir::failed(args)) {
    return mlir::failure();
  }

  if (argNames && argNames.size() > args->size()) {
    return op->emitOpError("more argument names than arguments");
  }

  return mlir::success();
}

/// Verify the sides of an equation or of an assignment. A single value of
/// the right-hand side can be implicitly converted from Integer to Real,
/// and so can the left-hand side of single equations. Outputs of function
/// calls are converted separately.
mlir::LogicalResult verifySides(mlir::Operation *op, mlir::ValueRange lhs,
                                mlir::ValueRange rhs, bool isEquation) {
  if (lhs.empty() || lhs.size() != rhs.size()) {
    return op->emitOpError("the two sides must have the same non-zero number "
                           "of values");
  }

  bool isSingle = lhs.size() == 1;

  for (auto [lhsValue, rhsValue] : llvm::zip(lhs, rhs)) {
    mlir::Type lhsType = lhsValue.getType();
    mlir::Type rhsType = rhsValue.getType();

    bool compatible = isCoercibleTo(rhsType, lhsType) ||
                      (isEquation && isSingle && isCoercibleTo(lhsType, rhsType));

    if (!compatible) {
      return op->emitOpError("incompatible types ")
             << lhsType << " and " << rhsType;
    }
  }

  return mlir::success();
}

/// Parse a for-equation or for-statement.
mlir::ParseResult parseForLoop(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  mlir::OpAsmParser::Argument inductionVariable;

  if (parser.parseArgument(inductionVariable, true)) {
    return mlir::failure();
  }

  mlir::Region *rangeRegion = result.addRegion();
  mlir::Region *bodyRegion = result.addRegion();

  if (mlir::failed(parser.parseOptionalKeyword("implicit"))) {
    if (parser.parseKeyword("in") || parser.parseRegion(*rangeRegion)) {
      return mlir::failure();
    }
  }

  if (parser.parseKeyword("do") ||
      parser.parseRegion(*bodyRegion, inductionVariable)) {
    return mlir::failure();
  }

  if (bodyRegion->empty()) {
    bodyRegion->emplaceBlock().addArgument(
        inductionVariable.type,
        parser.getEncodedSourceLoc(inductionVariable.ssaName.location));
  }

  return parser.parseOptionalAttrDictWithKeyword(result.attributes);
}

/// Print a for-equation or for-statement.
template <typename Op>
void printForLoop(mlir::OpAsmPrinter &printer, Op op) {
  mlir::BlockArgument inductionVariable = op.getInductionVariable();
  printer << " " << inductionVariable << " : " << inductionVariable.getType();

  if (op.hasImplicitRange()) {
    printer << " implicit";
  } else {
    printer << " in ";
    printer.printRegion(op.getRangeRegion());
  }

  printer << " do ";
  printer.printRegion(op.getBodyRegion(), false);
  printer.printOptionalAttrDictWithKeyword(op->getAttrs());
}

/// Verify a for-equation or for-statement.
template <typename Op>
mlir::LogicalResult verifyForLoop(Op op) {
  mlir::Block &body = op.getBodyRegion().front();

  if (body.getNumArguments() != 1) {
    return op.emitOpError("body must have exactly one argument, the loop "
                          "variable");
  }

  mlir::Type inductionType = body.getArgument(0).getType();

  if (op.hasImplicitRange()) {
    if (!mlir::isa<IntegerType, BooleanType, EnumerationType>(inductionType)) {
      return op.emitOpError("loop variables with implicit range must be "
                            "Integer, Boolean or enumeration scalars");
    }

    return mlir::success();
  }

  auto values = getYieldedValues(op, op.getRangeRegion(), "range", 1, 1);

  if (mlir::failed(values)) {
    return mlir::failure();
  }

  auto rangeType = mlir::dyn_cast<mlir::RankedTensorType>((*values)[0].getType());

  if (!rangeType || rangeType.getRank() != 1) {
    return op.emitOpError("range must be a vector, but got ")
           << (*values)[0].getType();
  }

  if (rangeType.getElementType() != inductionType) {
    return op.emitOpError("loop variable type ")
           << inductionType << " doesn't match the range type " << rangeType;
  }

  return mlir::success();
}

/// Verify that the region of an 'elsewhen' branch contains only the nested
/// when-equation or when-statement.
template <typename Op>
mlir::LogicalResult verifyElseWhen(mlir::Operation *op, mlir::Region &region) {
  if (region.empty()) {
    return mlir::success();
  }

  mlir::Block &block = region.front();

  if (block.getOperations().size() != 1 || !mlir::isa<Op>(block.front())) {
    return op->emitOpError("elsewhen region must contain exactly one '")
           << Op::getOperationName() << "'";
  }

  return mlir::success();
}

/// Get whether the operation is within a when-equation or when-statement,
/// excluding the case of being the 'elsewhen' branch of it.
template <typename Op>
bool isNestedInWhen(mlir::Operation *op) {
  mlir::Operation *current = op;

  while (mlir::Operation *parentOp = current->getParentOp()) {
    if (auto parentWhen = mlir::dyn_cast<Op>(parentOp)) {
      bool isElseWhen = mlir::isa<Op>(current) &&
                        current->getParentRegion() ==
                            &parentWhen.getElseWhenRegion();

      if (!isElseWhen) {
        return true;
      }
    }

    current = parentOp;
  }

  return false;
}
} // namespace

//===---------------------------------------------------------------------===//
// Equations
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::LogicalResult EquationOp::verifyRegions() {
  if (!mlir::isa<EquationSidesOp>(getBodyRegion().front().getTerminator())) {
    return emitOpError("body must be terminated by 'modelica.equation_sides'");
  }

  return mlir::success();
}

mlir::LogicalResult EquationSidesOp::verify() {
  return verifySides(getOperation(), getLhs(), getRhs(), true);
}

mlir::LogicalResult CallEquationOp::verifyRegions() {
  return verifyCallArguments(getOperation(), getBodyRegion(),
                             getArgNamesAttr());
}

mlir::ParseResult ForEquationOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  return parseForLoop(parser, result);
}

void ForEquationOp::print(mlir::OpAsmPrinter &printer) {
  printForLoop(printer, *this);
}

mlir::LogicalResult ForEquationOp::verifyRegions() {
  return verifyForLoop(*this);
}

mlir::LogicalResult IfEquationOp::verifyRegions() {
  return verifyCondition(getOperation(), getConditionRegion());
}

mlir::LogicalResult WhenEquationOp::verifyRegions() {
  // Restrictions on the usage of when-equations.
  if (auto section = getOperation()->getParentOfType<EquationSectionOp>();
      section && section.getIsInitial()) {
    return emitOpError("when-equations can't be used within initial "
                       "equations");
  }

  if (isNestedInWhen<WhenEquationOp>(getOperation())) {
    return emitOpError("when-equations can't be nested");
  }

  if (mlir::failed(verifyCondition(getOperation(), getConditionRegion(),
                                   true))) {
    return mlir::failure();
  }

  return verifyElseWhen<WhenEquationOp>(getOperation(), getElseWhenRegion());
}

mlir::LogicalResult ConnectOp::verifyRegions() {
  auto values = getYieldedValues(getOperation(), getBodyRegion(), "body", 2, 2);

  if (mlir::failed(values)) {
    return mlir::failure();
  }

  for (mlir::Value value : *values) {
    if (mlir::failed(verifyComponentReference(getOperation(), value,
                                              "connected element"))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Operators usable both as equations and as statements
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::LogicalResult ReinitOp::verifyRegions() {
  auto values = getYieldedValues(getOperation(), getBodyRegion(), "body", 2, 2);

  if (mlir::failed(values)) {
    return mlir::failure();
  }

  mlir::Value variable = (*values)[0];
  mlir::Value value = (*values)[1];

  if (mlir::failed(verifyComponentReference(getOperation(), variable,
                                            "reinitialized variable"))) {
    return mlir::failure();
  }

  if (!mlir::isa<RealType>(mlir::getElementTypeOrSelf(variable.getType()))) {
    return emitOpError("reinitialized variable must be Real, but got ")
           << variable.getType();
  }

  if (!isCoercibleTo(value.getType(), variable.getType())) {
    return emitOpError("value of type ")
           << value.getType() << " is not compatible with the variable type "
           << variable.getType();
  }

  return mlir::success();
}

mlir::LogicalResult AssertOp::verifyRegions() {
  auto values = getYieldedValues(getOperation(), getBodyRegion(), "body", 2, 3);

  if (mlir::failed(values)) {
    return mlir::failure();
  }

  if (!mlir::isa<BooleanType>((*values)[0].getType())) {
    return emitOpError("condition must be a Boolean scalar");
  }

  if (!mlir::isa<StringType>((*values)[1].getType())) {
    return emitOpError("message must be a String scalar");
  }

  if (values->size() == 3 &&
      !mlir::isa<EnumerationType>((*values)[2].getType())) {
    return emitOpError("level must be an AssertionLevel value");
  }

  return mlir::success();
}

mlir::LogicalResult TerminateOp::verifyRegions() {
  auto values = getYieldedValues(getOperation(), getBodyRegion(), "body", 1, 1);

  if (mlir::failed(values)) {
    return mlir::failure();
  }

  if (!mlir::isa<StringType>((*values)[0].getType())) {
    return emitOpError("message must be a String scalar");
  }

  return mlir::success();
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Equation operators for overconstrained connections
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::LogicalResult ConnectionsBranchOp::verifyRegions() {
  auto values = getYieldedValues(getOperation(), getBodyRegion(), "body", 2, 2);

  if (mlir::failed(values)) {
    return mlir::failure();
  }

  for (mlir::Value value : *values) {
    if (mlir::failed(verifyComponentReference(getOperation(), value, "node"))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult ConnectionsRootOp::verifyRegions() {
  auto values = getYieldedValues(getOperation(), getBodyRegion(), "body", 1, 1);

  if (mlir::failed(values)) {
    return mlir::failure();
  }

  return verifyComponentReference(getOperation(), (*values)[0], "node");
}

mlir::LogicalResult ConnectionsPotentialRootOp::verifyRegions() {
  auto values = getYieldedValues(getOperation(), getBodyRegion(), "body", 1, 2);

  if (mlir::failed(values)) {
    return mlir::failure();
  }

  if (mlir::failed(
          verifyComponentReference(getOperation(), (*values)[0], "node"))) {
    return mlir::failure();
  }

  if (values->size() == 2 && !mlir::isa<IntegerType>((*values)[1].getType())) {
    return emitOpError("priority must be an Integer scalar");
  }

  return mlir::success();
}

mlir::LogicalResult ConnectionsUniqueRootOp::verifyRegions() {
  auto values = getYieldedValues(getOperation(), getBodyRegion(), "body", 2, 2);

  if (mlir::failed(values)) {
    return mlir::failure();
  }

  if (mlir::failed(
          verifyComponentReference(getOperation(), (*values)[0], "node"))) {
    return mlir::failure();
  }

  if (!mlir::isa<StringType>((*values)[1].getType())) {
    return emitOpError("message must be a String scalar");
  }

  return mlir::success();
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Statements
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::LogicalResult AssignmentOp::verifyRegions() {
  if (!mlir::isa<AssignmentSidesOp>(getBodyRegion().front().getTerminator())) {
    return emitOpError(
        "body must be terminated by 'modelica.assignment_sides'");
  }

  return mlir::success();
}

mlir::LogicalResult AssignmentSidesOp::verify() {
  for (mlir::Value destination : getDestinations()) {
    if (mlir::failed(verifyComponentReference(getOperation(), destination,
                                              "destination", true))) {
      return mlir::failure();
    }
  }

  return verifySides(getOperation(), getDestinations(), getValues(), false);
}

mlir::LogicalResult CallStatementOp::verifyRegions() {
  return verifyCallArguments(getOperation(), getBodyRegion(),
                             getArgNamesAttr());
}

mlir::LogicalResult IfOp::verifyRegions() {
  return verifyCondition(getOperation(), getConditionRegion());
}

mlir::ParseResult ForOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseForLoop(parser, result);
}

void ForOp::print(mlir::OpAsmPrinter &printer) { printForLoop(printer, *this); }

mlir::LogicalResult ForOp::verifyRegions() { return verifyForLoop(*this); }

mlir::LogicalResult WhileOp::verifyRegions() {
  return verifyCondition(getOperation(), getConditionRegion());
}

mlir::LogicalResult WhenOp::verifyRegions() {
  // Restrictions on the usage of when-statements.
  mlir::Operation *parentOp = getOperation()->getParentOp();

  if (auto section = mlir::dyn_cast<AlgorithmSectionOp>(parentOp)) {
    if (section.getIsInitial()) {
      return emitOpError("when-statements can't be used within initial "
                         "algorithms");
    }

    if (auto classOp = section->getParentOfType<ClassOp>();
        classOp && (classOp.getKind() == ClassKind::function ||
                    classOp.getKind() == ClassKind::operator_function)) {
      return emitOpError("when-statements can't be used within functions");
    }
  } else if (isNestedInWhen<WhenOp>(getOperation())) {
    return emitOpError("when-statements can't be nested");
  } else if (!mlir::isa<WhenOp>(parentOp)) {
    return emitOpError("when-statements can't be used within while-loops, "
                       "for-loops, or if-statements");
  }

  if (mlir::failed(verifyCondition(getOperation(), getConditionRegion(),
                                   true))) {
    return mlir::failure();
  }

  return verifyElseWhen<WhenOp>(getOperation(), getElseWhenRegion());
}

mlir::LogicalResult BreakOp::verify() {
  mlir::Operation *parentOp = getOperation()->getParentOp();

  while (parentOp && !mlir::isa<AlgorithmSectionOp>(parentOp)) {
    if (mlir::isa<ForOp, WhileOp>(parentOp)) {
      return mlir::success();
    }

    parentOp = parentOp->getParentOp();
  }

  return emitOpError("must be within a for-loop or a while-loop");
}
} // namespace mlir::modelica
