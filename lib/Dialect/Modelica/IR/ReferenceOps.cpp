#include "marco/Dialect/Modelica/IR/OpUtils.h"
#include "marco/Dialect/Modelica/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/StringSet.h"

using namespace ::mlir::modelica;
using namespace ::mlir::modelica::detail;

//===---------------------------------------------------------------------===//
// MemberOp
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::LogicalResult MemberOp::verify() {
  mlir::Type objectType = getObject().getType();

  if (!mlir::isa<ObjectType>(mlir::getElementTypeOrSelf(objectType))) {
    return emitOpError("operand must be an object or an array of objects, "
                       "but got ")
           << objectType;
  }

  // Slice operation: the dimensions of the array of
  // objects are prepended to the ones of the member.
  llvm::ArrayRef<int64_t> objectShape = modelica::getShape(objectType);
  llvm::ArrayRef<int64_t> resultShape = modelica::getShape(getType());

  if (resultShape.size() < objectShape.size()) {
    return emitOpError("result must have at least the dimensions of the "
                       "array of objects");
  }

  for (auto [objectDim, resultDim] : llvm::zip(objectShape, resultShape)) {
    if (!mlir::ShapedType::isDynamic(objectDim) &&
        !mlir::ShapedType::isDynamic(resultDim) && objectDim != resultDim) {
      return emitOpError("result dimensions don't match the dimensions of "
                         "the array of objects");
    }
  }

  return mlir::success();
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// SubscriptOp
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
void SubscriptOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Type resultType, mlir::Value array,
                        mlir::ValueRange indices) {
  llvm::SmallVector<bool> colons(indices.size(), false);
  build(builder, state, resultType, array, indices,
        builder.getDenseBoolArrayAttr(colons));
}

mlir::ParseResult SubscriptOp::parse(mlir::OpAsmParser &parser,
                                     mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand array;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> indices;
  llvm::SmallVector<bool> colons;

  auto parseSubscript = [&]() -> mlir::ParseResult {
    if (mlir::succeeded(parser.parseOptionalColon())) {
      colons.push_back(true);
      return mlir::success();
    }

    colons.push_back(false);
    return parser.parseOperand(indices.emplace_back());
  };

  mlir::FunctionType functionType;
  llvm::SMLoc typesLoc;

  if (parser.parseOperand(array) ||
      parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Square,
                                     parseSubscript) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon()) {
    return mlir::failure();
  }

  typesLoc = parser.getCurrentLocation();

  if (parser.parseType(functionType)) {
    return mlir::failure();
  }

  if (functionType.getNumInputs() != indices.size() + 1 ||
      functionType.getNumResults() != 1) {
    return parser.emitError(typesLoc, "expected the types of the array and of ")
           << indices.size() << " indices, and one result type";
  }

  result.addAttribute(getColonsAttrName(result.name),
                      parser.getBuilder().getDenseBoolArrayAttr(colons));

  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
  operands.push_back(array);
  llvm::append_range(operands, indices);

  if (parser.resolveOperands(operands, functionType.getInputs(), typesLoc,
                             result.operands)) {
    return mlir::failure();
  }

  result.addTypes(functionType.getResults());
  return mlir::success();
}

void SubscriptOp::print(mlir::OpAsmPrinter &printer) {
  printer << " " << getArray() << "[";

  auto indices = getIndices().begin();
  bool first = true;

  for (bool isColon : getColons()) {
    if (!first) {
      printer << ", ";
    }

    first = false;

    if (isColon) {
      printer << ":";
    } else {
      printer << *indices++;
    }
  }

  printer << "]";
  printer.printOptionalAttrDict((*this)->getAttrs(), {getColonsAttrName()});
  printer << " : ";
  printer.printFunctionalType(getOperation());
}

mlir::LogicalResult SubscriptOp::verify() {
  auto arrayType = mlir::cast<mlir::RankedTensorType>(getArray().getType());
  llvm::ArrayRef<bool> colons = getColons();

  size_t numIndices = llvm::count(colons, false);

  if (numIndices != getIndices().size()) {
    return emitOpError("expected ")
           << numIndices << " index operands, but got "
           << getIndices().size();
  }

  if (static_cast<int64_t>(colons.size()) > arrayType.getRank()) {
    return emitOpError("too many subscripts for an array of rank ")
           << arrayType.getRank();
  }

  // Compute the shape of the result.
  llvm::SmallVector<int64_t> shape;
  auto indices = getIndices().begin();

  for (auto [dim, isColon] : llvm::enumerate(colons)) {
    if (isColon) {
      shape.push_back(arrayType.getDimSize(dim));
      continue;
    }

    mlir::Type indexType = (*indices++).getType();
    mlir::Type indexElementType = mlir::getElementTypeOrSelf(indexType);

    if (!mlir::isa<IntegerType, BooleanType, EnumerationType>(
            indexElementType)) {
      return emitOpError("indices must be Integer, Boolean or enumeration "
                         "values, but got ")
             << indexType;
    }

    if (auto indexTensorType =
            mlir::dyn_cast<mlir::RankedTensorType>(indexType)) {
      if (indexTensorType.getRank() != 1) {
        return emitOpError("array indices must be vectors, but got ")
               << indexType;
      }

      shape.push_back(indexTensorType.getDimSize(0));
    }
  }

  // Trailing dimensions are selected entirely.
  for (int64_t dim = colons.size(); dim < arrayType.getRank(); ++dim) {
    shape.push_back(arrayType.getDimSize(dim));
  }

  mlir::Type expectedType = getValueType(shape, arrayType.getElementType());

  if (!areCompatibleTypes(expectedType, getType())) {
    return emitOpError("expected result type compatible with ")
           << expectedType << ", but got " << getType();
  }

  return mlir::success();
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// CallOp
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
size_t CallOp::getNumPositionalArgs() {
  size_t numNamedArgs = getArgNamesAttr() ? getArgNamesAttr().size() : 0;
  return getArgs().size() - numNamedArgs;
}

mlir::ParseResult CallOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  mlir::SymbolRefAttr callee;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> args;
  llvm::SmallVector<mlir::Attribute> argNames;

  auto parseArgument = [&]() -> mlir::ParseResult {
    mlir::StringAttr name;

    if (mlir::succeeded(parser.parseOptionalSymbolName(name))) {
      if (parser.parseEqual()) {
        return mlir::failure();
      }

      argNames.push_back(name);
    } else if (!argNames.empty()) {
      return parser.emitError(parser.getCurrentLocation(),
                              "positional arguments must precede the named "
                              "ones");
    }

    return parser.parseOperand(args.emplace_back());
  };

  if (parser.parseAttribute(callee, getCalleeAttrName(result.name),
                            result.attributes) ||
      parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Paren,
                                     parseArgument) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon()) {
    return mlir::failure();
  }

  llvm::SMLoc typesLoc = parser.getCurrentLocation();
  mlir::FunctionType functionType;

  if (parser.parseType(functionType) ||
      parser.resolveOperands(args, functionType.getInputs(), typesLoc,
                             result.operands)) {
    return mlir::failure();
  }

  if (!argNames.empty()) {
    result.addAttribute(getArgNamesAttrName(result.name),
                        parser.getBuilder().getArrayAttr(argNames));
  }

  result.addTypes(functionType.getResults());
  return mlir::success();
}

void CallOp::print(mlir::OpAsmPrinter &printer) {
  printer << " " << getCalleeAttr() << "(";

  size_t numPositionalArgs = getNumPositionalArgs();

  for (auto [index, arg] : llvm::enumerate(getArgs())) {
    if (index != 0) {
      printer << ", ";
    }

    if (index >= numPositionalArgs) {
      printer.printSymbolName(
          mlir::cast<mlir::StringAttr>(
              getArgNamesAttr()[index - numPositionalArgs])
              .getValue());

      printer << " = ";
    }

    printer << arg;
  }

  printer << ")";

  printer.printOptionalAttrDict((*this)->getAttrs(),
                                {getCalleeAttrName(), getArgNamesAttrName()});

  printer << " : ";
  printer.printFunctionalType(getArgs().getTypes(), getResultTypes());
}

mlir::LogicalResult CallOp::verify() {
  mlir::ArrayAttr argNames = getArgNamesAttr();

  if (!argNames) {
    return mlir::success();
  }

  if (argNames.size() > getArgs().size()) {
    return emitOpError("more argument names than arguments");
  }

  llvm::StringSet<> names;

  for (mlir::Attribute name : argNames) {
    if (!names.insert(mlir::cast<mlir::StringAttr>(name).getValue()).second) {
      return emitOpError("duplicate named argument ") << name;
    }
  }

  return mlir::success();
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Operators on connectors
//===---------------------------------------------------------------------===//

namespace {
/// Verify the operand of a stream operator.
mlir::LogicalResult verifyStreamOperand(mlir::Operation *op,
                                        mlir::Value operand) {
  if (mlir::failed(
          verifyComponentReference(op, operand, "stream variable"))) {
    return mlir::failure();
  }

  if (!mlir::isa<RealType>(mlir::getElementTypeOrSelf(operand.getType()))) {
    return op->emitOpError("stream variable must be Real, but got ")
           << operand.getType();
  }

  return mlir::success();
}
} // namespace

namespace mlir::modelica {
mlir::LogicalResult InStreamOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferSameAsOperand(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool InStreamOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                         mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult InStreamOp::verify() {
  return verifyStreamOperand(getOperation(), getOperand());
}

mlir::LogicalResult ActualStreamOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferSameAsOperand(context, location, adaptor.getOperands(),
                                    inferredReturnTypes);
}

bool ActualStreamOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                             mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult ActualStreamOp::verify() {
  return verifyStreamOperand(getOperation(), getOperand());
}

mlir::LogicalResult CardinalityOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferCardinality(context, location, adaptor.getOperands(),
                                  inferredReturnTypes);
}

bool CardinalityOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                            mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult CardinalityOp::verify() {
  return verifyComponentReference(getOperation(), getOperand(), "connector");
}

mlir::LogicalResult SpatialDistributionOp::verify() {
  for (mlir::Value value : {getIn0(), getIn1(), getX()}) {
    if (!isNumericType(value.getType())) {
      return emitOpError("in0, in1 and x must be numeric scalars");
    }
  }

  for (mlir::Value value : {getInitialPoints(), getInitialValues()}) {
    if (!value) {
      continue;
    }

    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());

    if (!tensorType || tensorType.getRank() != 1 ||
        !isNumericType(tensorType.getElementType())) {
      return emitOpError("initial points and values must be numeric "
                         "vectors");
    }
  }

  return mlir::success();
}

mlir::LogicalResult GetInstanceNameOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferGetInstanceName(context, location, adaptor.getOperands(),
                                      inferredReturnTypes);
}

bool GetInstanceNameOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                                mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult ConnectionsIsRootOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferConnectionsPredicate(
      context, location, adaptor.getOperands(), inferredReturnTypes);
}

bool ConnectionsIsRootOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                                  mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult ConnectionsIsRootOp::verify() {
  return verifyComponentReference(getOperation(), getOperand(), "node");
}

mlir::LogicalResult ConnectionsRootedOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferConnectionsPredicate(
      context, location, adaptor.getOperands(), inferredReturnTypes);
}

bool ConnectionsRootedOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                                  mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult ConnectionsRootedOp::verify() {
  return verifyComponentReference(getOperation(), getOperand(), "node");
}

mlir::LogicalResult ConnectionsUniqueRootIndicesOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return detail::inferConnectionsUniqueRootIndices(
      context, location, adaptor.getOperands(), inferredReturnTypes);
}

bool ConnectionsUniqueRootIndicesOp::isCompatibleReturnTypes(
    mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return areCompatibleTypes(lhs, rhs);
}

mlir::LogicalResult ConnectionsUniqueRootIndicesOp::verify() {
  if (mlir::failed(
          verifyComponentReference(getOperation(), getRoots(), "roots"))) {
    return mlir::failure();
  }

  return verifyComponentReference(getOperation(), getNodes(), "nodes");
}
} // namespace mlir::modelica
