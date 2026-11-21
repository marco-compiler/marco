#include "marco/Dialect/Modelica/IR/Ops.h"
#include "marco/Dialect/Modelica/IR/ConstantFolding.h"
#include "marco/Dialect/Modelica/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include <cmath>

using namespace ::mlir::modelica;

#define GET_OP_CLASSES
#include "marco/Dialect/Modelica/IR/Ops.cpp.inc"

//===---------------------------------------------------------------------===//
// Modelica Dialect
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
void ModelicaDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "marco/Dialect/Modelica/IR/Ops.cpp.inc"
      >();
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Folding utilities
//===---------------------------------------------------------------------===//

namespace {
using IntegerFn = llvm::function_ref<std::optional<int64_t>(int64_t, int64_t)>;
using RealFn = llvm::function_ref<std::optional<double>(double, double)>;

/// Fold a binary numeric operation on scalar literals. The Integer function
/// is used when the result is Integer, and the Real one when the result is
/// Real. Unitful literals are not folded, as the units would be lost.
mlir::Attribute foldNumeric(mlir::Type elementType, mlir::Attribute lhs,
                            mlir::Attribute rhs, IntegerFn integerFn,
                            RealFn realFn) {
  if (hasUnit(lhs) || hasUnit(rhs)) {
    return {};
  }

  if (mlir::isa<IntegerType>(elementType)) {
    auto x = getIntegerValue(lhs);
    auto y = getIntegerValue(rhs);

    if (!x || !y) {
      return {};
    }

    if (auto result = integerFn(*x, *y)) {
      return IntegerAttr::get(elementType.getContext(), elementType, *result,
                              mlir::StringAttr());
    }

    return {};
  }

  if (mlir::isa<RealType>(elementType)) {
    auto x = getRealValue(lhs);
    auto y = getRealValue(rhs);

    if (!x || !y) {
      return {};
    }

    if (auto result = realFn(*x, *y)) {
      return getNumericAttr(elementType, *result);
    }
  }

  return {};
}

/// Fold an element-wise binary numeric operation.
mlir::OpFoldResult foldNumericElementwise(mlir::Type resultType,
                                          mlir::Attribute lhs,
                                          mlir::Attribute rhs,
                                          IntegerFn integerFn, RealFn realFn) {
  return foldElementwise(
      resultType, {lhs, rhs},
      [&](mlir::Type elementType, llvm::ArrayRef<mlir::Attribute> operands) {
        return foldNumeric(elementType, operands[0], operands[1], integerFn,
                           realFn);
      });
}

/// Fold an element-wise unary numeric operation. The Real function is
/// applied to the operand converted to Real, and its result is converted to
/// the result type. If the operand is Integer and an Integer function is
/// given, the Integer function is used instead.
mlir::OpFoldResult foldNumericUnary(
    mlir::Type resultType, mlir::Attribute operand,
    llvm::function_ref<std::optional<double>(double)> realFn,
    llvm::function_ref<std::optional<int64_t>(int64_t)> integerFn = nullptr) {
  return foldElementwise(
      resultType, operand,
      [&](mlir::Type elementType,
          llvm::ArrayRef<mlir::Attribute> operands) -> mlir::Attribute {
        if (hasUnit(operands[0])) {
          return {};
        }

        if (integerFn && mlir::isa<IntegerType>(elementType)) {
          if (auto x = getIntegerValue(operands[0])) {
            if (auto result = integerFn(*x)) {
              return IntegerAttr::get(elementType.getContext(), elementType,
                                      *result, mlir::StringAttr());
            }

            return {};
          }
        }

        auto x = getRealValue(operands[0]);

        if (!x) {
          return {};
        }

        if (auto result = realFn(*x)) {
          return getNumericAttr(elementType, *result);
        }

        return {};
      });
}

std::optional<int64_t> checkedAdd(int64_t x, int64_t y) {
  int64_t result;

  if (__builtin_add_overflow(x, y, &result)) {
    return std::nullopt;
  }

  return result;
}

std::optional<int64_t> checkedSub(int64_t x, int64_t y) {
  int64_t result;

  if (__builtin_sub_overflow(x, y, &result)) {
    return std::nullopt;
  }

  return result;
}

std::optional<int64_t> checkedMul(int64_t x, int64_t y) {
  int64_t result;

  if (__builtin_mul_overflow(x, y, &result)) {
    return std::nullopt;
  }

  return result;
}

std::optional<double> realAdd(double x, double y) { return x + y; }

std::optional<double> realSub(double x, double y) { return x - y; }

std::optional<double> realMul(double x, double y) { return x * y; }

std::optional<double> realDiv(double x, double y) {
  if (y == 0) {
    return std::nullopt;
  }

  return x / y;
}

std::optional<int64_t> noInteger(int64_t, int64_t) { return std::nullopt; }

/// Exponentiation of scalars.
mlir::Attribute foldPow(mlir::Type elementType, mlir::Attribute base,
                        mlir::Attribute exponent) {
  if (hasUnit(base) || hasUnit(exponent)) {
    return {};
  }

  auto x = getRealValue(base);

  if (!x) {
    return {};
  }

  if (auto n = getIntegerValue(exponent)) {
    if (*n == 0) {
      return getNumericAttr(elementType, 1);
    }

    if (*x == 0 && *n < 0) {
      return {};
    }

    return getNumericAttr(elementType, std::pow(*x, static_cast<double>(*n)));
  }

  auto y = getRealValue(exponent);

  // Negative bases with Real exponents are not folded, so that the deprecated
  // semantics for Real exponents with integer values is not relied upon.
  if (!y || *x < 0 || (*x == 0 && *y <= 0)) {
    return {};
  }

  return getNumericAttr(elementType, std::pow(*x, *y));
}

/// String concatenation of scalar literals.
mlir::Attribute foldConcatenation(mlir::Type elementType, mlir::Attribute lhs,
                                  mlir::Attribute rhs) {
  auto x = mlir::dyn_cast<StringValueAttr>(lhs);
  auto y = mlir::dyn_cast<StringValueAttr>(rhs);

  if (!x || !y) {
    return {};
  }

  return StringValueAttr::get(elementType.getContext(), elementType,
                              (x.getValue() + y.getValue()).str());
}

/// Fold an addition, which can be either numeric or a String concatenation.
mlir::OpFoldResult foldAddition(mlir::Type resultType, mlir::Attribute lhs,
                                mlir::Attribute rhs) {
  return foldElementwise(
      resultType, {lhs, rhs},
      [&](mlir::Type elementType, llvm::ArrayRef<mlir::Attribute> operands) {
        if (mlir::isa<StringType>(elementType)) {
          return foldConcatenation(elementType, operands[0], operands[1]);
        }

        return foldNumeric(elementType, operands[0], operands[1], checkedAdd,
                           realAdd);
      });
}

/// Fold a relational operator, given a predicate on the result of the
/// comparison.
mlir::OpFoldResult foldRelational(mlir::Type resultType, mlir::Attribute lhs,
                                  mlir::Attribute rhs,
                                  llvm::function_ref<bool(int)> predicate) {
  if (auto comparison = compareScalars(lhs, rhs)) {
    return BooleanAttr::get(resultType.getContext(), resultType,
                            predicate(*comparison));
  }

  return {};
}

/// Fold an element-wise Boolean operation.
mlir::OpFoldResult
foldBoolean(mlir::Type resultType, llvm::ArrayRef<mlir::Attribute> operands,
            llvm::function_ref<bool(llvm::ArrayRef<bool>)> fn) {
  return foldElementwise(
      resultType, operands,
      [&](mlir::Type elementType,
          llvm::ArrayRef<mlir::Attribute> scalars) -> mlir::Attribute {
        llvm::SmallVector<bool, 2> values;

        for (mlir::Attribute scalar : scalars) {
          auto value = getBooleanValue(scalar);

          if (!value) {
            return {};
          }

          values.push_back(*value);
        }

        return BooleanAttr::get(elementType.getContext(), elementType,
                                fn(values));
      });
}

/// Fold the binary min and max operators.
mlir::OpFoldResult foldMinMax(mlir::Type resultType, mlir::Attribute lhs,
                              mlir::Attribute rhs, bool isMin) {
  return foldElementwise(
      resultType, {lhs, rhs},
      [&](mlir::Type elementType,
          llvm::ArrayRef<mlir::Attribute> operands) -> mlir::Attribute {
        auto comparison = compareScalars(operands[0], operands[1]);

        if (!comparison) {
          return {};
        }

        bool takeLhs = isMin ? *comparison <= 0 : *comparison >= 0;
        return convertScalarAttr(takeLhs ? operands[0] : operands[1],
                                 elementType);
      });
}

/// Verify the iterators of an array constructor or of a reduction
/// expression, and get the type of the value yielded by the body.
mlir::FailureOr<mlir::Type> verifyIterators(mlir::Operation *op,
                                            mlir::ValueRange ranges,
                                            mlir::Region &body) {
  if (ranges.empty()) {
    return op->emitOpError("requires at least one iterator");
  }

  mlir::Block &block = body.front();

  if (block.getNumArguments() != ranges.size()) {
    return op->emitOpError("expected ")
           << ranges.size() << " block arguments, but got "
           << block.getNumArguments();
  }

  for (auto [range, arg] : llvm::zip(ranges, block.getArguments())) {
    auto rangeType = mlir::cast<mlir::RankedTensorType>(range.getType());

    if (rangeType.getRank() != 1) {
      return op->emitOpError("iteration ranges must be vectors, but got ")
             << rangeType;
    }

    if (arg.getType() != rangeType.getElementType()) {
      return op->emitOpError("iterator type ")
             << arg.getType() << " doesn't match the element type of range "
             << rangeType;
    }
  }

  auto yieldOp = mlir::dyn_cast<YieldOp>(block.getTerminator());

  if (!yieldOp || yieldOp.getValues().size() != 1) {
    return op->emitOpError("body must yield exactly one value");
  }

  return yieldOp.getValues()[0].getType();
}

/// Print the iterators of an array constructor or of a reduction
/// expression, together with the body.
void printIterators(mlir::OpAsmPrinter &printer, mlir::Operation *op,
                    mlir::ValueRange ranges, mlir::Region &body,
                    llvm::ArrayRef<llvm::StringRef> elidedAttrs = {}) {
  printer << " (";

  llvm::interleaveComma(
      llvm::zip(body.getArguments(), ranges), printer, [&](auto pair) {
        auto [arg, range] = pair;
        printer << arg << " in " << range << " : " << range.getType();
      });

  printer << ") -> " << op->getResultTypes()[0] << " ";
  printer.printRegion(body, false);
  printer.printOptionalAttrDictWithKeyword(op->getAttrs(), elidedAttrs);
}

/// Parse the iterators of an array constructor or of a reduction
/// expression, together with the body.
mlir::ParseResult parseIterators(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::Argument> args;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> ranges;
  llvm::SmallVector<mlir::Type> rangeTypes;
  llvm::SMLoc rangesLoc = parser.getCurrentLocation();

  auto parseIterator = [&]() -> mlir::ParseResult {
    mlir::OpAsmParser::Argument &arg = args.emplace_back();
    mlir::OpAsmParser::UnresolvedOperand &range = ranges.emplace_back();
    mlir::Type &rangeType = rangeTypes.emplace_back();

    if (parser.parseArgument(arg) || parser.parseKeyword("in") ||
        parser.parseOperand(range) || parser.parseColonType(rangeType)) {
      return mlir::failure();
    }

    arg.type = mlir::getElementTypeOrSelf(rangeType);
    return mlir::success();
  };

  mlir::Type resultType;
  mlir::Region *body = result.addRegion();

  if (parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Paren,
                                     parseIterator) ||
      parser.parseArrow() || parser.parseType(resultType) ||
      parser.parseRegion(*body, args) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parser.resolveOperands(ranges, rangeTypes, rangesLoc,
                             result.operands)) {
    return mlir::failure();
  }

  result.addTypes(resultType);
  return mlir::success();
}
} // namespace

//===---------------------------------------------------------------------===//
// ConstantOp
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::LogicalResult ConstantOp::verify() {
  if (!mlir::isa<BooleanAttr, IntegerAttr, RealAttr, StringValueAttr,
                 EnumLiteralAttr, ArrayValueAttr>(getValue())) {
    return emitOpError("value must be a literal of the Modelica dialect");
  }

  return mlir::success();
}

mlir::OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return getValue();
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Arithmetic operators
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::LogicalResult NegateOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericUnary(context, location, adaptor.getOperands(),
                                   inferredReturnTypes);
}

bool NegateOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                       mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult NegateOp::fold(FoldAdaptor adaptor) {
  // Negated unitful literals keep their unit, so that implicit unit
  // conversions can be applied to them.
  return foldElementwise(
      getType(), adaptor.getOperand(),
      [](mlir::Type elementType,
         llvm::ArrayRef<mlir::Attribute> operands) -> mlir::Attribute {
        if (auto x = mlir::dyn_cast<IntegerAttr>(operands[0])) {
          if (auto result = checkedSub(0, x.getValue())) {
            return IntegerAttr::get(elementType.getContext(), elementType,
                                    *result, x.getUnit());
          }

          return {};
        }

        if (auto x = mlir::dyn_cast<RealAttr>(operands[0])) {
          return RealAttr::get(elementType.getContext(), elementType,
                               -x.getValue(), x.getUnit());
        }

        return {};
      });
}

mlir::LogicalResult AddOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferAdd(context, location, adaptor.getOperands(),
                          inferredReturnTypes);
}

bool AddOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  return foldAddition(getType(), adaptor.getLhs(), adaptor.getRhs());
}

mlir::LogicalResult AddEWOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferAddEW(context, location, adaptor.getOperands(),
                            inferredReturnTypes);
}

bool AddEWOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                      mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult AddEWOp::fold(FoldAdaptor adaptor) {
  return foldAddition(getType(), adaptor.getLhs(), adaptor.getRhs());
}

mlir::LogicalResult SubOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferSub(context, location, adaptor.getOperands(),
                          inferredReturnTypes);
}

bool SubOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  return foldNumericElementwise(getType(), adaptor.getLhs(), adaptor.getRhs(),
                                checkedSub, realSub);
}

mlir::LogicalResult SubEWOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericBroadcast(context, location, adaptor.getOperands(),
                                       inferredReturnTypes);
}

bool SubEWOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                      mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult SubEWOp::fold(FoldAdaptor adaptor) {
  return foldNumericElementwise(getType(), adaptor.getLhs(), adaptor.getRhs(),
                                checkedSub, realSub);
}

mlir::LogicalResult MulOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferMul(context, location, adaptor.getOperands(),
                          inferredReturnTypes);
}

bool MulOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  // Only the scalar multiplication is folded.
  if (!isScalarType(getLhs().getType()) && !isScalarType(getRhs().getType())) {
    return {};
  }

  return foldNumericElementwise(getType(), adaptor.getLhs(), adaptor.getRhs(),
                                checkedMul, realMul);
}

mlir::LogicalResult MulEWOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericBroadcast(context, location, adaptor.getOperands(),
                                       inferredReturnTypes);
}

bool MulEWOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                      mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult MulEWOp::fold(FoldAdaptor adaptor) {
  return foldNumericElementwise(getType(), adaptor.getLhs(), adaptor.getRhs(),
                                checkedMul, realMul);
}

mlir::LogicalResult DivOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferDiv(context, location, adaptor.getOperands(),
                          inferredReturnTypes);
}

bool DivOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult DivOp::fold(FoldAdaptor adaptor) {
  return foldNumericElementwise(getType(), adaptor.getLhs(), adaptor.getRhs(),
                                noInteger, realDiv);
}

mlir::LogicalResult DivEWOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferRealBroadcast(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool DivEWOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                      mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult DivEWOp::fold(FoldAdaptor adaptor) {
  return foldNumericElementwise(getType(), adaptor.getLhs(), adaptor.getRhs(),
                                noInteger, realDiv);
}

mlir::LogicalResult PowOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferPow(context, location, adaptor.getOperands(),
                          inferredReturnTypes);
}

bool PowOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult PowOp::fold(FoldAdaptor adaptor) {
  // The exponentiation of matrices is not folded.
  if (!isScalarType(getLhs().getType())) {
    return {};
  }

  if (!adaptor.getLhs() || !adaptor.getRhs()) {
    return {};
  }

  return foldPow(getType(), adaptor.getLhs(), adaptor.getRhs());
}

mlir::LogicalResult PowEWOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferRealBroadcast(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool PowEWOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                      mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult PowEWOp::fold(FoldAdaptor adaptor) {
  return foldElementwise(
      getType(), {adaptor.getLhs(), adaptor.getRhs()},
      [](mlir::Type elementType, llvm::ArrayRef<mlir::Attribute> operands) {
        return foldPow(elementType, operands[0], operands[1]);
      });
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Relational and logical operators
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::LogicalResult
EqOp::inferReturnTypes(mlir::MLIRContext *context,
                       std::optional<mlir::Location> location, Adaptor adaptor,
                       llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferRelational(context, location, adaptor.getOperands(),
                                 inferredReturnTypes);
}

bool EqOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult EqOp::fold(FoldAdaptor adaptor) {
  return foldRelational(getType(), adaptor.getLhs(), adaptor.getRhs(),
                        [](int cmp) { return cmp == 0; });
}

mlir::LogicalResult NotEqOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferRelational(context, location, adaptor.getOperands(),
                                 inferredReturnTypes);
}

bool NotEqOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                      mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult NotEqOp::fold(FoldAdaptor adaptor) {
  return foldRelational(getType(), adaptor.getLhs(), adaptor.getRhs(),
                        [](int cmp) { return cmp != 0; });
}

mlir::LogicalResult
LtOp::inferReturnTypes(mlir::MLIRContext *context,
                       std::optional<mlir::Location> location, Adaptor adaptor,
                       llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferRelational(context, location, adaptor.getOperands(),
                                 inferredReturnTypes);
}

bool LtOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult LtOp::fold(FoldAdaptor adaptor) {
  return foldRelational(getType(), adaptor.getLhs(), adaptor.getRhs(),
                        [](int cmp) { return cmp < 0; });
}

mlir::LogicalResult LteOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferRelational(context, location, adaptor.getOperands(),
                                 inferredReturnTypes);
}

bool LteOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult LteOp::fold(FoldAdaptor adaptor) {
  return foldRelational(getType(), adaptor.getLhs(), adaptor.getRhs(),
                        [](int cmp) { return cmp <= 0; });
}

mlir::LogicalResult
GtOp::inferReturnTypes(mlir::MLIRContext *context,
                       std::optional<mlir::Location> location, Adaptor adaptor,
                       llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferRelational(context, location, adaptor.getOperands(),
                                 inferredReturnTypes);
}

bool GtOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult GtOp::fold(FoldAdaptor adaptor) {
  return foldRelational(getType(), adaptor.getLhs(), adaptor.getRhs(),
                        [](int cmp) { return cmp > 0; });
}

mlir::LogicalResult GteOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferRelational(context, location, adaptor.getOperands(),
                                 inferredReturnTypes);
}

bool GteOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult GteOp::fold(FoldAdaptor adaptor) {
  return foldRelational(getType(), adaptor.getLhs(), adaptor.getRhs(),
                        [](int cmp) { return cmp >= 0; });
}

mlir::LogicalResult NotOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferBooleanUnary(context, location, adaptor.getOperands(),
                                   inferredReturnTypes);
}

bool NotOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult NotOp::fold(FoldAdaptor adaptor) {
  return foldBoolean(getType(), adaptor.getOperand(),
                     [](llvm::ArrayRef<bool> values) { return !values[0]; });
}

mlir::LogicalResult AndOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferBooleanBinary(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool AndOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult AndOp::fold(FoldAdaptor adaptor) {
  return foldBoolean(getType(), {adaptor.getLhs(), adaptor.getRhs()},
                     [](llvm::ArrayRef<bool> values) {
                       return values[0] && values[1];
                     });
}

mlir::LogicalResult
OrOp::inferReturnTypes(mlir::MLIRContext *context,
                       std::optional<mlir::Location> location, Adaptor adaptor,
                       llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferBooleanBinary(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool OrOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult OrOp::fold(FoldAdaptor adaptor) {
  return foldBoolean(getType(), {adaptor.getLhs(), adaptor.getRhs()},
                     [](llvm::ArrayRef<bool> values) {
                       return values[0] || values[1];
                     });
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// IfExprOp
//===---------------------------------------------------------------------===//

namespace {
/// Replace an if-expression having a constant condition with the content of
/// the selected region.
struct IfExprConstantConditionPattern
    : public mlir::OpRewritePattern<IfExprOp> {
  using mlir::OpRewritePattern<IfExprOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(IfExprOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::Attribute condition;

    if (!mlir::matchPattern(op.getCondition(), mlir::m_Constant(&condition))) {
      return mlir::failure();
    }

    auto conditionValue = getBooleanValue(condition);

    if (!conditionValue) {
      return mlir::failure();
    }

    mlir::Region &region =
        *conditionValue ? op.getThenRegion() : op.getElseRegion();

    mlir::Block *block = &region.front();
    auto yieldOp = mlir::cast<YieldOp>(block->getTerminator());
    mlir::Value yieldedValue = yieldOp.getValues()[0];

    // The yielded value may need an implicit conversion to the result type,
    // which can't be represented without the if-expression.
    if (yieldedValue.getType() != op.getType()) {
      return mlir::failure();
    }

    rewriter.inlineBlockBefore(block, op);
    rewriter.replaceOp(op, yieldedValue);
    rewriter.eraseOp(yieldOp);
    return mlir::success();
  }
};
} // namespace

namespace mlir::modelica {
mlir::LogicalResult IfExprOp::verify() {
  for (mlir::Region *region : getRegions()) {
    auto yieldOp = mlir::dyn_cast<YieldOp>(region->front().getTerminator());

    if (!yieldOp || yieldOp.getValues().size() != 1) {
      return emitOpError("regions must yield exactly one value");
    }

    mlir::Type yieldedType = yieldOp.getValues()[0].getType();

    if (!isCoercibleTo(yieldedType, getType())) {
      return emitOpError("yielded type ")
             << yieldedType << " is not compatible with the result type "
             << getType();
    }
  }

  return mlir::success();
}

void IfExprOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                           mlir::MLIRContext *context) {
  patterns.add<IfExprConstantConditionPattern>(context);
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Array constructors
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::LogicalResult RangeOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferRange(context, location, adaptor.getOperands(),
                            inferredReturnTypes);
}

bool RangeOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                      mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult RangeOp::fold(FoldAdaptor adaptor) {
  auto resultType = mlir::cast<mlir::RankedTensorType>(getType());

  if (!resultType.hasStaticShape() || !adaptor.getStart() ||
      !adaptor.getStop() || (getStep() && !adaptor.getStep())) {
    return {};
  }

  auto size = getRangeSize(adaptor.getStart(), adaptor.getStop(),
                           adaptor.getStep());

  if (!size || *size != resultType.getDimSize(0)) {
    return {};
  }

  mlir::Type elementType = resultType.getElementType();
  llvm::SmallVector<mlir::Attribute> elements;

  if (mlir::isa<IntegerType>(elementType)) {
    int64_t start = *getIntegerValue(adaptor.getStart());
    int64_t step = getStep() ? *getIntegerValue(adaptor.getStep()) : 1;

    for (int64_t i = 0; i < *size; ++i) {
      elements.push_back(IntegerAttr::get(getContext(), elementType,
                                          start + i * step,
                                          mlir::StringAttr()));
    }
  } else if (mlir::isa<RealType>(elementType)) {
    double start = *getRealValue(adaptor.getStart());
    double step = getStep() ? *getRealValue(adaptor.getStep()) : 1;

    for (int64_t i = 0; i < *size; ++i) {
      elements.push_back(
          getNumericAttr(elementType, start + static_cast<double>(i) * step));
    }
  } else if (mlir::isa<BooleanType>(elementType)) {
    bool start = *getBooleanValue(adaptor.getStart());

    for (int64_t i = 0; i < *size; ++i) {
      elements.push_back(
          BooleanAttr::get(getContext(), elementType, start || i > 0));
    }
  } else {
    // Enumeration ranges can't be folded, as the names of the literals are
    // not known.
    return {};
  }

  return ArrayValueAttr::get(getContext(), resultType, elements);
}

mlir::LogicalResult ArrayOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferArray(context, location, adaptor.getOperands(),
                            inferredReturnTypes);
}

bool ArrayOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                      mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult ArrayOp::fold(FoldAdaptor adaptor) {
  auto resultType = mlir::cast<mlir::RankedTensorType>(getType());

  if (!resultType.hasStaticShape()) {
    return {};
  }

  llvm::SmallVector<mlir::Attribute> elements;

  for (mlir::Attribute operand : adaptor.getValues()) {
    if (!operand) {
      return {};
    }

    for (mlir::Attribute element : getConstantElements(operand)) {
      mlir::Attribute converted =
          convertScalarAttr(element, resultType.getElementType());

      if (!converted) {
        return {};
      }

      elements.push_back(converted);
    }
  }

  if (static_cast<int64_t>(elements.size()) != resultType.getNumElements()) {
    return {};
  }

  return ArrayValueAttr::get(getContext(), resultType, elements);
}

mlir::LogicalResult PromoteOp::verify() {
  llvm::ArrayRef<int64_t> operandShape = getShape(getOperand().getType());
  int64_t rank = getRank();

  if (rank < static_cast<int64_t>(operandShape.size())) {
    return emitOpError("rank must be greater than or equal to the rank of "
                       "the operand");
  }

  llvm::SmallVector<int64_t> expectedShape(operandShape);
  expectedShape.resize(rank, 1);

  mlir::Type expectedType = getValueType(
      expectedShape, mlir::getElementTypeOrSelf(getOperand().getType()));

  if (!areCompatibleTypes(expectedType, getType())) {
    return emitOpError("expected result type ")
           << expectedType << ", but got " << getType();
  }

  return mlir::success();
}

mlir::LogicalResult CatOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferCat(context, location, adaptor.getOperands(),
                          inferredReturnTypes);
}

bool CatOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult IdentityOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferIdentity(context, location, adaptor.getOperands(),
                               inferredReturnTypes);
}

bool IdentityOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                         mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult DiagonalOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferDiagonal(context, location, adaptor.getOperands(),
                               inferredReturnTypes);
}

bool DiagonalOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                         mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult ZerosOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferIntegerFill(context, location, adaptor.getOperands(),
                                  inferredReturnTypes);
}

bool ZerosOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                      mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult OnesOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferIntegerFill(context, location, adaptor.getOperands(),
                                  inferredReturnTypes);
}

bool OnesOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult FillOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferFill(context, location, adaptor.getOperands(),
                           inferredReturnTypes);
}

bool FillOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult LinspaceOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferLinspace(context, location, adaptor.getOperands(),
                               inferredReturnTypes);
}

bool LinspaceOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                         mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Iterator-based expressions
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::ParseResult ArrayForOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  return parseIterators(parser, result);
}

void ArrayForOp::print(mlir::OpAsmPrinter &printer) {
  printIterators(printer, getOperation(), getRanges(), getBody());
}

mlir::LogicalResult ArrayForOp::verify() {
  auto yieldedType = verifyIterators(getOperation(), getRanges(), getBody());

  if (mlir::failed(yieldedType)) {
    return mlir::failure();
  }

  // The last iterator varies along the outermost dimension.
  llvm::SmallVector<int64_t> expectedShape;

  for (mlir::Value range : llvm::reverse(getRanges())) {
    expectedShape.push_back(
        mlir::cast<mlir::RankedTensorType>(range.getType()).getDimSize(0));
  }

  llvm::append_range(expectedShape, getShape(*yieldedType));

  mlir::Type expectedType =
      getValueType(expectedShape, mlir::getElementTypeOrSelf(*yieldedType));

  if (!isCoercibleTo(expectedType, getType())) {
    return emitOpError("expected result type compatible with ")
           << expectedType << ", but got " << getType();
  }

  return mlir::success();
}

mlir::ParseResult ReductionOp::parse(mlir::OpAsmParser &parser,
                                     mlir::OperationState &result) {
  llvm::StringRef kindKeyword;
  llvm::SMLoc kindLoc = parser.getCurrentLocation();

  if (parser.parseKeyword(&kindKeyword)) {
    return mlir::failure();
  }

  auto kind = symbolizeReductionKind(kindKeyword);

  if (!kind) {
    return parser.emitError(kindLoc, "unknown reduction kind: ")
           << kindKeyword;
  }

  result.getOrAddProperties<Properties>().kind =
      ReductionKindAttr::get(parser.getContext(), *kind);

  return parseIterators(parser, result);
}

void ReductionOp::print(mlir::OpAsmPrinter &printer) {
  printer << " " << stringifyReductionKind(getKind());
  printIterators(printer, getOperation(), getRanges(), getBody(),
                 {getKindAttrName()});
}

/// Verify the type of the elements being reduced.
static mlir::LogicalResult verifyReducedType(mlir::Operation *op,
                                             ReductionKind kind,
                                             mlir::Type type) {
  mlir::Type elementType = mlir::getElementTypeOrSelf(type);
  bool isScalar = isScalarType(type);

  switch (kind) {
  case ReductionKind::sum:
    if (!isNumericType(elementType)) {
      return op->emitOpError("sum requires Integer or Real values, but got ")
             << type;
    }

    break;

  case ReductionKind::product:
    if (!isScalar || !isNumericType(elementType)) {
      return op->emitOpError("product requires scalar Integer or Real "
                             "values, but got ")
             << type;
    }

    break;

  case ReductionKind::min:
  case ReductionKind::max:
    if (!isScalar ||
        !mlir::isa<IntegerType, RealType, BooleanType, EnumerationType>(
            elementType)) {
      return op->emitOpError("min and max require scalar enumeration, "
                             "Boolean, Integer or Real values, but got ")
             << type;
    }

    break;
  }

  return mlir::success();
}

mlir::LogicalResult ReductionOp::verify() {
  auto yieldedType = verifyIterators(getOperation(), getRanges(), getBody());

  if (mlir::failed(yieldedType)) {
    return mlir::failure();
  }

  if (mlir::failed(verifyReducedType(getOperation(), getKind(), *yieldedType))) {
    return mlir::failure();
  }

  if (!isCoercibleTo(*yieldedType, getType())) {
    return emitOpError("expected result type compatible with ")
           << *yieldedType << ", but got " << getType();
  }

  return mlir::success();
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Array dimension and conversion functions
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::LogicalResult NDimsOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNDims(context, location, adaptor.getOperands(),
                            inferredReturnTypes);
}

bool NDimsOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                      mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult NDimsOp::fold(FoldAdaptor adaptor) {
  return IntegerAttr::get(
      getContext(), getType(),
      static_cast<int64_t>(modelica::getShape(getOperand().getType()).size()),
      mlir::StringAttr());
}

mlir::LogicalResult SizeOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferSize(context, location, adaptor.getOperands(),
                           inferredReturnTypes);
}

bool SizeOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult SizeOp::fold(FoldAdaptor adaptor) {
  llvm::ArrayRef<int64_t> shape = modelica::getShape(getArray().getType());

  if (getDim()) {
    auto dim = getIntegerValue(adaptor.getDim());

    if (!dim || *dim < 0 || *dim >= static_cast<int64_t>(shape.size()) ||
        mlir::ShapedType::isDynamic(shape[*dim])) {
      return {};
    }

    return IntegerAttr::get(getContext(), getType(), shape[*dim],
                            mlir::StringAttr());
  }

  if (mlir::ShapedType::isDynamicShape(shape)) {
    return {};
  }

  auto resultType = mlir::cast<mlir::RankedTensorType>(getType());
  llvm::SmallVector<mlir::Attribute> elements;

  for (int64_t dim : shape) {
    elements.push_back(IntegerAttr::get(
        getContext(), resultType.getElementType(), dim, mlir::StringAttr()));
  }

  return ArrayValueAttr::get(getContext(), resultType, elements);
}

mlir::LogicalResult ScalarOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferScalar(context, location, adaptor.getOperands(),
                             inferredReturnTypes);
}

bool ScalarOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                       mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult ScalarOp::fold(FoldAdaptor adaptor) {
  auto arrayAttr = mlir::dyn_cast_or_null<ArrayValueAttr>(adaptor.getOperand());

  if (!arrayAttr || arrayAttr.getElements().size() != 1) {
    return {};
  }

  return arrayAttr.getElements()[0];
}

mlir::LogicalResult VectorOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferVector(context, location, adaptor.getOperands(),
                             inferredReturnTypes);
}

bool VectorOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                       mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult MatrixOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferMatrix(context, location, adaptor.getOperands(),
                             inferredReturnTypes);
}

bool MatrixOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                       mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// ArrayReduceOp
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::LogicalResult ArrayReduceOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferArrayReduce(context, location, adaptor.getOperands(),
                                  inferredReturnTypes);
}

bool ArrayReduceOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                            mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult ArrayReduceOp::verify() {
  return verifyReducedType(getOperation(), getKind(), getType());
}

mlir::OpFoldResult ArrayReduceOp::fold(FoldAdaptor adaptor) {
  auto arrayAttr = mlir::dyn_cast_or_null<ArrayValueAttr>(adaptor.getArray());

  if (!arrayAttr) {
    return {};
  }

  llvm::ArrayRef<mlir::Attribute> elements = arrayAttr.getElements();
  mlir::Type resultType = getType();

  if (llvm::any_of(elements, hasUnit)) {
    return {};
  }

  switch (getKind()) {
  case ReductionKind::sum:
  case ReductionKind::product: {
    bool isSum = getKind() == ReductionKind::sum;

    if (mlir::isa<IntegerType>(resultType)) {
      std::optional<int64_t> result = isSum ? 0 : 1;

      for (mlir::Attribute element : elements) {
        auto value = getIntegerValue(element);

        if (!value || !result) {
          return {};
        }

        result = isSum ? checkedAdd(*result, *value)
                       : checkedMul(*result, *value);
      }

      if (!result) {
        return {};
      }

      return IntegerAttr::get(getContext(), resultType, *result,
                              mlir::StringAttr());
    }

    double result = isSum ? 0 : 1;

    for (mlir::Attribute element : elements) {
      auto value = getRealValue(element);

      if (!value) {
        return {};
      }

      result = isSum ? result + *value : result * *value;
    }

    return getNumericAttr(resultType, result);
  }

  case ReductionKind::min:
  case ReductionKind::max: {
    // The result for empty arrays depends on the limits of the type, which
    // are left unspecified by the dialect.
    if (elements.empty()) {
      return {};
    }

    bool isMin = getKind() == ReductionKind::min;
    mlir::Attribute result = elements.front();

    for (mlir::Attribute element : elements.drop_front()) {
      auto comparison = compareScalars(element, result);

      if (!comparison) {
        return {};
      }

      if (isMin ? *comparison < 0 : *comparison > 0) {
        result = element;
      }
    }

    return result;
  }
  }

  return {};
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Matrix and vector algebra functions
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::LogicalResult TransposeOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferTranspose(context, location, adaptor.getOperands(),
                                inferredReturnTypes);
}

bool TransposeOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                          mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult OuterProductOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferOuterProduct(context, location, adaptor.getOperands(),
                                   inferredReturnTypes);
}

bool OuterProductOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                             mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult SymmetricOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferSymmetric(context, location, adaptor.getOperands(),
                                inferredReturnTypes);
}

bool SymmetricOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                          mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult CrossOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferCross(context, location, adaptor.getOperands(),
                            inferredReturnTypes);
}

bool CrossOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                      mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult SkewOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferSkew(context, location, adaptor.getOperands(),
                           inferredReturnTypes);
}

bool SkewOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Numeric functions and conversion operators
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::LogicalResult AbsOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericUnary(context, location, adaptor.getOperands(),
                                   inferredReturnTypes);
}

bool AbsOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult AbsOp::fold(FoldAdaptor adaptor) {
  return foldNumericUnary(
      getType(), adaptor.getOperand(),
      [](double x) -> std::optional<double> { return std::abs(x); },
      [](int64_t x) -> std::optional<int64_t> {
        if (x == std::numeric_limits<int64_t>::min()) {
          return std::nullopt;
        }

        return x < 0 ? -x : x;
      });
}

mlir::LogicalResult SignOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericToInteger(context, location, adaptor.getOperands(),
                                       inferredReturnTypes);
}

bool SignOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult SignOp::fold(FoldAdaptor adaptor) {
  return foldNumericUnary(getType(), adaptor.getOperand(),
                          [](double x) -> std::optional<double> {
                            return (x > 0) - (x < 0);
                          });
}

mlir::LogicalResult SqrtOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericToReal(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool SqrtOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult SqrtOp::fold(FoldAdaptor adaptor) {
  return foldNumericUnary(getType(), adaptor.getOperand(),
                          [](double x) -> std::optional<double> {
                            if (x < 0) {
                              return std::nullopt;
                            }

                            return std::sqrt(x);
                          });
}

mlir::LogicalResult NthRootOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNthRoot(context, location, adaptor.getOperands(),
                              inferredReturnTypes);
}

bool NthRootOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                        mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult NthRootOp::fold(FoldAdaptor adaptor) {
  auto n = getIntegerValue(adaptor.getN());

  if (!n || *n <= 0) {
    return {};
  }

  return foldNumericUnary(getType(), adaptor.getOperand(),
                          [&](double x) -> std::optional<double> {
                            bool isEven = *n % 2 == 0;

                            if (isEven && x < 0) {
                              return std::nullopt;
                            }

                            double root = std::pow(
                                std::abs(x), 1.0 / static_cast<double>(*n));

                            return x < 0 ? -root : root;
                          });
}

mlir::LogicalResult EnumToIntegerOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferEnumToInteger(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool EnumToIntegerOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                              mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult EnumToIntegerOp::fold(FoldAdaptor adaptor) {
  return foldElementwise(
      getType(), adaptor.getOperand(),
      [](mlir::Type elementType,
         llvm::ArrayRef<mlir::Attribute> operands) -> mlir::Attribute {
        if (auto literal = mlir::dyn_cast<EnumLiteralAttr>(operands[0])) {
          return IntegerAttr::get(elementType.getContext(), elementType,
                                  literal.getOrdinal(), mlir::StringAttr());
        }

        return {};
      });
}

mlir::LogicalResult IntegerToEnumOp::verify() {
  mlir::Type operandType = getOperand().getType();
  mlir::Type resultType = getType();

  if (!mlir::isa<IntegerType>(mlir::getElementTypeOrSelf(operandType))) {
    return emitOpError("operand must be Integer, but got ") << operandType;
  }

  auto enumType =
      mlir::dyn_cast<EnumerationType>(mlir::getElementTypeOrSelf(resultType));

  if (!enumType) {
    return emitOpError("result must be an enumeration, but got ")
           << resultType;
  }

  if (!areCompatibleTypes(getValueType(modelica::getShape(operandType), enumType),
                          resultType)) {
    return emitOpError("result shape doesn't match the operand shape");
  }

  return mlir::success();
}

mlir::LogicalResult ToStringOp::verify() {
  mlir::Type valueType = getValue().getType();

  if (!mlir::isa<BooleanType, IntegerType, RealType, EnumerationType>(
          valueType)) {
    return emitOpError("value must be a Boolean, Integer, Real or "
                       "enumeration scalar, but got ")
           << valueType;
  }

  if (getFormat() &&
      (getMinimumLength() || getLeftJustified() || getSignificantDigits())) {
    return emitOpError("format can't be combined with the other options");
  }

  if (getSignificantDigits() && !mlir::isa<RealType>(valueType)) {
    return emitOpError("significantDigits is allowed only for Real values");
  }

  if (getFormat() && !mlir::isa<IntegerType, RealType>(valueType)) {
    return emitOpError("format is allowed only for Integer and Real values");
  }

  return mlir::success();
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Event-triggering mathematical functions
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::LogicalResult DivTruncOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericBroadcast(context, location, adaptor.getOperands(),
                                       inferredReturnTypes);
}

bool DivTruncOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                         mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult DivTruncOp::fold(FoldAdaptor adaptor) {
  return foldNumericElementwise(
      getType(), adaptor.getLhs(), adaptor.getRhs(),
      [](int64_t x, int64_t y) -> std::optional<int64_t> {
        if (y == 0 || (x == std::numeric_limits<int64_t>::min() && y == -1)) {
          return std::nullopt;
        }

        return x / y;
      },
      [](double x, double y) -> std::optional<double> {
        if (y == 0) {
          return std::nullopt;
        }

        return std::trunc(x / y);
      });
}

mlir::LogicalResult ModOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericBroadcast(context, location, adaptor.getOperands(),
                                       inferredReturnTypes);
}

bool ModOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult ModOp::fold(FoldAdaptor adaptor) {
  return foldNumericElementwise(
      getType(), adaptor.getLhs(), adaptor.getRhs(),
      [](int64_t x, int64_t y) -> std::optional<int64_t> {
        if (y == 0 || (x == std::numeric_limits<int64_t>::min() && y == -1)) {
          return std::nullopt;
        }

        int64_t result = x % y;

        if (result != 0 && ((result < 0) != (y < 0))) {
          result += y;
        }

        return result;
      },
      [](double x, double y) -> std::optional<double> {
        if (y == 0) {
          return std::nullopt;
        }

        return x - std::floor(x / y) * y;
      });
}

mlir::LogicalResult RemOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericBroadcast(context, location, adaptor.getOperands(),
                                       inferredReturnTypes);
}

bool RemOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult RemOp::fold(FoldAdaptor adaptor) {
  return foldNumericElementwise(
      getType(), adaptor.getLhs(), adaptor.getRhs(),
      [](int64_t x, int64_t y) -> std::optional<int64_t> {
        if (y == 0 || (x == std::numeric_limits<int64_t>::min() && y == -1)) {
          return std::nullopt;
        }

        return x % y;
      },
      [](double x, double y) -> std::optional<double> {
        if (y == 0) {
          return std::nullopt;
        }

        return std::fmod(x, y);
      });
}

mlir::LogicalResult CeilOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericToReal(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool CeilOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult CeilOp::fold(FoldAdaptor adaptor) {
  return foldNumericUnary(
      getType(), adaptor.getOperand(),
      [](double x) -> std::optional<double> { return std::ceil(x); });
}

mlir::LogicalResult FloorOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericToReal(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool FloorOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                      mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult FloorOp::fold(FoldAdaptor adaptor) {
  return foldNumericUnary(
      getType(), adaptor.getOperand(),
      [](double x) -> std::optional<double> { return std::floor(x); });
}

mlir::LogicalResult IntegerOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericToInteger(context, location, adaptor.getOperands(),
                                       inferredReturnTypes);
}

bool IntegerOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                        mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult IntegerOp::fold(FoldAdaptor adaptor) {
  return foldNumericUnary(
      getType(), adaptor.getOperand(),
      [](double x) -> std::optional<double> { return std::floor(x); },
      [](int64_t x) -> std::optional<int64_t> { return x; });
}

mlir::LogicalResult DelayOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferDelay(context, location, adaptor.getOperands(),
                            inferredReturnTypes);
}

bool DelayOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                      mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Elementary mathematical functions
//===---------------------------------------------------------------------===//

namespace {
/// Fold an elementary mathematical function, given its domain.
mlir::OpFoldResult
foldElementary(mlir::Type resultType, mlir::Attribute operand,
               double (*fn)(double),
               llvm::function_ref<bool(double)> domain = nullptr) {
  return foldNumericUnary(resultType, operand,
                          [&](double x) -> std::optional<double> {
                            if (domain && !domain(x)) {
                              return std::nullopt;
                            }

                            return fn(x);
                          });
}

bool isInUnitInterval(double x) { return x >= -1 && x <= 1; }

bool isPositive(double x) { return x > 0; }
} // namespace

namespace mlir::modelica {
mlir::LogicalResult SinOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericToReal(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool SinOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult SinOp::fold(FoldAdaptor adaptor) {
  return foldElementary(getType(), adaptor.getOperand(), std::sin);
}

mlir::LogicalResult CosOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericToReal(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool CosOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult CosOp::fold(FoldAdaptor adaptor) {
  return foldElementary(getType(), adaptor.getOperand(), std::cos);
}

mlir::LogicalResult TanOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericToReal(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool TanOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult TanOp::fold(FoldAdaptor adaptor) {
  return foldElementary(getType(), adaptor.getOperand(), std::tan);
}

mlir::LogicalResult AsinOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericToReal(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool AsinOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult AsinOp::fold(FoldAdaptor adaptor) {
  return foldElementary(getType(), adaptor.getOperand(), std::asin,
                        isInUnitInterval);
}

mlir::LogicalResult AcosOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericToReal(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool AcosOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult AcosOp::fold(FoldAdaptor adaptor) {
  return foldElementary(getType(), adaptor.getOperand(), std::acos,
                        isInUnitInterval);
}

mlir::LogicalResult AtanOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericToReal(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool AtanOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult AtanOp::fold(FoldAdaptor adaptor) {
  return foldElementary(getType(), adaptor.getOperand(), std::atan);
}

mlir::LogicalResult SinhOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericToReal(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool SinhOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult SinhOp::fold(FoldAdaptor adaptor) {
  return foldElementary(getType(), adaptor.getOperand(), std::sinh);
}

mlir::LogicalResult CoshOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericToReal(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool CoshOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult CoshOp::fold(FoldAdaptor adaptor) {
  return foldElementary(getType(), adaptor.getOperand(), std::cosh);
}

mlir::LogicalResult TanhOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericToReal(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool TanhOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult TanhOp::fold(FoldAdaptor adaptor) {
  return foldElementary(getType(), adaptor.getOperand(), std::tanh);
}

mlir::LogicalResult ExpOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericToReal(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool ExpOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult ExpOp::fold(FoldAdaptor adaptor) {
  return foldElementary(getType(), adaptor.getOperand(), std::exp);
}

mlir::LogicalResult LogOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericToReal(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool LogOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult LogOp::fold(FoldAdaptor adaptor) {
  return foldElementary(getType(), adaptor.getOperand(), std::log,
                        isPositive);
}

mlir::LogicalResult Log10Op::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferNumericToReal(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool Log10Op::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                      mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult Log10Op::fold(FoldAdaptor adaptor) {
  return foldElementary(getType(), adaptor.getOperand(), std::log10,
                        isPositive);
}

mlir::LogicalResult Atan2Op::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferRealBroadcast(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool Atan2Op::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                      mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult Atan2Op::fold(FoldAdaptor adaptor) {
  return foldNumericElementwise(
      getType(), adaptor.getLhs(), adaptor.getRhs(), noInteger,
      [](double y, double x) -> std::optional<double> {
        return std::atan2(y, x);
      });
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Derivative and special purpose operators
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::LogicalResult DerOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferDer(context, location, adaptor.getOperands(),
                          inferredReturnTypes);
}

bool DerOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult HomotopyOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferHomotopy(context, location, adaptor.getOperands(),
                               inferredReturnTypes);
}

bool HomotopyOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                         mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult SemiLinearOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferSemiLinear(context, location, adaptor.getOperands(),
                                 inferredReturnTypes);
}

bool SemiLinearOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                           mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Event-related operators
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::LogicalResult TimeOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferTime(context, location, adaptor.getOperands(),
                           inferredReturnTypes);
}

bool TimeOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult InitialOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferBooleanNullary(context, location, adaptor.getOperands(),
                                     inferredReturnTypes);
}

bool InitialOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                        mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult TerminalOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferBooleanNullary(context, location, adaptor.getOperands(),
                                     inferredReturnTypes);
}

bool TerminalOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                         mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult NoEventOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferSameAsOperand(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool NoEventOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                        mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult NoEventOp::fold(FoldAdaptor adaptor) {
  return adaptor.getOperand();
}

mlir::LogicalResult SmoothOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferSmooth(context, location, adaptor.getOperands(),
                             inferredReturnTypes);
}

bool SmoothOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                       mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult SmoothOp::fold(FoldAdaptor adaptor) {
  return adaptor.getExpr();
}

mlir::LogicalResult SampleOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferSample(context, location, adaptor.getOperands(),
                             inferredReturnTypes);
}

bool SampleOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                       mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult PreOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferSameAsOperand(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool PreOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult EdgeOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferBooleanUnary(context, location, adaptor.getOperands(),
                                   inferredReturnTypes);
}

bool EdgeOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Binary min and max
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::LogicalResult ChangeOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferChange(context, location, adaptor.getOperands(),
                             inferredReturnTypes);
}

bool ChangeOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                       mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult MinOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferMinMax(context, location, adaptor.getOperands(),
                             inferredReturnTypes);
}

bool MinOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult MinOp::fold(FoldAdaptor adaptor) {
  return foldMinMax(getType(), adaptor.getLhs(), adaptor.getRhs(), true);
}

mlir::LogicalResult MaxOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferMinMax(context, location, adaptor.getOperands(),
                             inferredReturnTypes);
}

bool MaxOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::OpFoldResult MaxOp::fold(FoldAdaptor adaptor) {
  return foldMinMax(getType(), adaptor.getLhs(), adaptor.getRhs(), false);
}
} // namespace mlir::modelica
