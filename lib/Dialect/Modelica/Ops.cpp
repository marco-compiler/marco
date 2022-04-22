#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/Modelica/Ops.h"
#include "marco/Codegen/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include <cmath>

using namespace ::mlir::modelica;

static mlir::Value readValue(mlir::OpBuilder& builder, mlir::Value operand)
{
  if (auto arrayType = operand.getType().dyn_cast<ArrayType>(); arrayType && arrayType.getRank() == 0) {
    return builder.create<LoadOp>(operand.getLoc(), operand);
  }

  return operand;
}

static mlir::Type convertToRealType(mlir::Type type)
{
  if (auto arrayType = type.dyn_cast<ArrayType>()) {
    return arrayType.toElementType(RealType::get(type.getContext()));
  }

  return RealType::get(type.getContext());
}

static bool isScalar(mlir::Type type)
{
  if (!type) {
    return false;
  }

  return type.isa<BooleanType, IntegerType, RealType, mlir::IndexType>();
}

static bool isScalar(mlir::Attribute attribute)
{
  if (!attribute) {
    return false;
  }

  return isScalar(attribute.getType());
}

static bool isScalarIntegerLike(mlir::Type type)
{
  if (!isScalar(type)) {
    return false;
  }

  return type.isa<BooleanType, IntegerType, mlir::IndexType>();
}

static bool isScalarIntegerLike(mlir::Attribute attribute)
{
  if (!attribute) {
    return false;
  }

  return isScalarIntegerLike(attribute.getType());
}

static bool isScalarFloatLike(mlir::Type type)
{
  if (!isScalar(type)) {
    return false;
  }

  return type.isa<RealType>();
}

static bool isScalarFloatLike(mlir::Attribute attribute)
{
  if (!attribute) {
    return false;
  }

  return isScalarFloatLike(attribute.getType());
}

static long getScalarIntegerLikeValue(mlir::Attribute attribute)
{
  assert(isScalarIntegerLike(attribute));

  if (auto booleanAttr = attribute.dyn_cast<BooleanAttr>()) {
    return booleanAttr.getValue();
  }

  if (auto integerAttr = attribute.dyn_cast<IntegerAttr>()) {
    return integerAttr.getValue().getSExtValue();
  }

  return attribute.dyn_cast<mlir::IntegerAttr>().getValue().getSExtValue();
}

static double getScalarFloatLikeValue(mlir::Attribute attribute)
{
  assert(isScalarFloatLike(attribute));
  return attribute.dyn_cast<RealAttr>().getValue().convertToDouble();
}

static mlir::LogicalResult verify(AbsOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(AcosOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(AddOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(AddEWOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(AllocOp op)
{
  auto dynamicDimensionsAmount = op.getArrayType().getDynamicDimensionsCount();
  auto valuesAmount = op.dynamicSizes().size();

  if (dynamicDimensionsAmount != valuesAmount) {
    return op.emitOpError(
        "incorrect number of values for dynamic dimensions (expected " +
        std::to_string(dynamicDimensionsAmount) + ", got " +
        std::to_string(valuesAmount) + ")");
  }

  return mlir::success();
}

static mlir::LogicalResult verify(AllocaOp op)
{
  auto dynamicDimensionsAmount = op.getArrayType().getDynamicDimensionsCount();
  auto valuesAmount = op.dynamicSizes().size();

  if (dynamicDimensionsAmount != valuesAmount) {
    return op.emitOpError(
        "incorrect number of values for dynamic dimensions (expected " +
            std::to_string(dynamicDimensionsAmount) + ", got " +
            std::to_string(valuesAmount) + ")");
  }

  return mlir::success();
}

static mlir::LogicalResult verify(AndOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(ArrayCastOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(AsinOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(AtanOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(Atan2Op op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(ConstantOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(CosOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(CoshOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(DerOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(DiagonalOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(DimOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(DivOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(DivEWOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(ExpOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(ArrayFillOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(FreeOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(IdentityOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(LinspaceOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(LoadOp op)
{
  auto indicesAmount = op.indices().size();
  auto rank = op.getArrayType().getRank();

  if (indicesAmount != rank) {
    return op.emitOpError(
        "incorrect number of indices for load (expected " +
        std::to_string(rank) + ", got " + std::to_string(indicesAmount) + ")");
  }

  return mlir::success();
}

static mlir::LogicalResult verify(SubscriptionOp op)
{
  auto indicesAmount = op.indices().size();

  if (op.getSourceArrayType().slice(indicesAmount) != op.getResultArrayType()) {
    return op.emitOpError("incompatible source array type and result sliced type");
  }

  return mlir::success();
}

static mlir::LogicalResult verify(LogOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(Log10Op op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(MulOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(MulEWOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(NegateOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(NotOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(OnesOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(OrOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(PowOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(PowEWOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(PrintOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(ProductOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(SignOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(SinOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(SinhOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(SizeOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(SqrtOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(StoreOp op)
{
  auto indicesAmount = op.indices().size();
  auto rank = op.getArrayType().getRank();

  if (indicesAmount != rank) {
    return op.emitOpError(
        "incorrect number of indices for store (expected " +
        std::to_string(rank) + ", got " + std::to_string(indicesAmount) + ")");
  }

  return mlir::success();
}

static mlir::LogicalResult verify(SubOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(SubEWOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(SumOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(SymmetricOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(TanOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(TanhOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(TransposeOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(ZerosOp op)
{
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseConstantOp(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
  mlir::Attribute value;

  if (parser.parseAttribute(value)) {
    return mlir::failure();
  }

  result.attributes.append("value", value);
  result.addTypes(value.getType());

  return mlir::success();
}

static void print(mlir::OpAsmPrinter& printer, ConstantOp op)
{
  printer << op.getOperationName();
  printer.printOptionalAttrDict(op->getAttrs(), {"value"});
  printer << " " << op.value();

  // If the value is a symbol reference, print a trailing type.
  if (op.value().isa<mlir::SymbolRefAttr>()) {
    printer << " : " << op.getType();
  }
}

//===----------------------------------------------------------------------===//
// MemberCreateOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseMemberCreateOp(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
  auto& builder = parser.getBuilder();
  mlir::StringAttr nameAttr;

  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes)) {
    return mlir::failure();
  }

  llvm::SmallVector<mlir::OpAsmParser::OperandType, 1> dynamicSizes;
  mlir::Type resultType;

  if (parser.parseOperandList(dynamicSizes) ||
      parser.resolveOperands(dynamicSizes, builder.getIndexType(), result.operands) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon() ||
      parser.parseType(resultType)) {
    return mlir::failure();
  }

  result.addTypes(resultType);
  return mlir::success();
}

static void print(mlir::OpAsmPrinter& printer, MemberCreateOp op)
{
  printer << op.getOperationName() << " ";
  printer.printSymbolName(op.name());
  printer.printOptionalAttrDict(op->getAttrs(), { "sym_name" });
  printer << op.dynamicSizes();
  printer << " : " << op.getResult().getType();
}

//===----------------------------------------------------------------------===//
// MemberLoadOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseMemberLoadOp(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
  mlir::OpAsmParser::OperandType member;
  mlir::Type memberType;
  mlir::Type resultType;

  if (parser.parseOperand(member) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(memberType) ||
      parser.resolveOperand(member, memberType, result.operands)) {
    return mlir::failure();
  }

  if (mlir::succeeded(parser.parseOptionalArrow())) {
    if (parser.parseType(resultType)) {
      return mlir::failure();
    }

    result.addTypes(resultType);
  } else {
    result.addTypes(memberType.cast<MemberType>().unwrap());
  }

  return mlir::success();
}

static void print(mlir::OpAsmPrinter& printer, MemberLoadOp op)
{
  printer << op.getOperationName();
  printer << " " << op.member();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : " << op.member().getType();

  if (auto resultType = op.getResult().getType(); resultType != op.getMemberType().unwrap()) {
    printer << " -> " << resultType;
  }
}

//===----------------------------------------------------------------------===//
// MemberStoreOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseMemberStoreOp(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
  mlir::OpAsmParser::OperandType member;
  mlir::OpAsmParser::OperandType value;
  mlir::Type memberType;
  mlir::Type valueType;

  if (parser.parseOperand(member) ||
      parser.parseComma() ||
      parser.parseOperand(value) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(memberType) ||
      parser.resolveOperand(member, memberType, result.operands)) {
    return mlir::failure();
  }

  if (mlir::succeeded(parser.parseOptionalComma())) {
    if (parser.parseType(valueType) ||
        parser.resolveOperand(value, valueType, result.operands)) {
      return mlir::failure();
    }
  } else if (parser.resolveOperand(value, memberType.cast<MemberType>().unwrap(), result.operands)) {
    return mlir::failure();
  }

  return mlir::success();
}

static void print(mlir::OpAsmPrinter& printer, MemberStoreOp op)
{
  printer << op.getOperationName();
  printer << " " << op.member() << ", " << op.value();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : " << op.member().getType();

  if (auto valueType = op.value().getType(); valueType != op.getMemberType().unwrap()) {
    printer << ", " << valueType;
  }
}

//===----------------------------------------------------------------------===//
// ModelOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseModelOp(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
    return mlir::failure();
  }

  mlir::Region* initRegion = result.addRegion();
  mlir::Region* bodyRegion = result.addRegion();

  if (parser.parseRegion(*initRegion) ||
      parser.parseKeyword("equations") ||
      parser.parseRegion(*bodyRegion)) {
    return mlir::failure();
  }

  return mlir::success();
}

static void print(mlir::OpAsmPrinter& printer, ModelOp op)
{
  printer << op.getOperationName();
  printer.printOptionalAttrDictWithKeyword(op->getAttrs());
  printer.printRegion(op.initRegion());
  printer << " equations";
  printer.printRegion(op.bodyRegion());
}

//===----------------------------------------------------------------------===//
// FunctionOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseFunctionOp(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
  auto& builder = parser.getBuilder();
  mlir::StringAttr nameAttr;

  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes)) {
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Type, 3> argsTypes;
  llvm::SmallVector<mlir::Type, 3> resultsTypes;

  if (parser.parseColon() ||
      parser.parseLParen()) {
    return mlir::failure();
  }

  if (mlir::failed(parser.parseOptionalRParen())) {
    if (parser.parseTypeList(argsTypes) ||
        parser.parseRParen()) {
      return mlir::failure();
    }
  }

  if (parser.parseArrow() ||
      parser.parseLParen()) {
    return mlir::failure();
  }

  if (mlir::failed(parser.parseOptionalRParen())) {
    if (parser.parseTypeList(resultsTypes) ||
        parser.parseRParen()) {
      return mlir::failure();
    }
  }

  auto functionType = builder.getFunctionType(argsTypes, resultsTypes);
  result.attributes.append("type", mlir::TypeAttr::get(functionType));

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
    return mlir::failure();
  }

  mlir::Region* bodyRegion = result.addRegion();

  if (parser.parseRegion(*bodyRegion, llvm::None, llvm::None)) {
    return mlir::failure();
  }

  return mlir::success();
}

static void print(mlir::OpAsmPrinter& printer, FunctionOp op)
{
  printer << op.getOperationName() << " ";
  printer.printSymbolName(op.name());
  printer << " : " << op.getType();
  printer.printOptionalAttrDictWithKeyword(op->getAttrs(), { "sym_name", "type" });
  printer.printRegion(op.body());
}

//===----------------------------------------------------------------------===//
// DerFunctionOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseDerFunctionOp(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
  mlir::StringAttr nameAttr;

  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return mlir::failure();
  }

  return mlir::success();
}

static void print(mlir::OpAsmPrinter& printer, DerFunctionOp op)
{
  printer << op.getOperationName() << " ";
  printer.printSymbolName(op.name());
  printer.printOptionalAttrDict(op->getAttrs(), { "sym_name" });
}

//===----------------------------------------------------------------------===//
// ForEquationOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseForEquationOp(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
  auto& builder = parser.getBuilder();

  mlir::OpAsmParser::OperandType induction;
  long from;
  long to;

  if (parser.parseOperand(induction) ||
      parser.resolveOperand(induction, builder.getIndexType(), result.operands) ||
      parser.parseEqual() ||
      parser.parseInteger(from) ||
      parser.parseKeyword("to") ||
      parser.parseInteger(to)) {
    return mlir::failure();
  }

  result.attributes.append("from", builder.getIndexAttr(from));
  result.attributes.append("to", builder.getIndexAttr(to));

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return mlir::failure();
  }

  mlir::Region* bodyRegion = result.addRegion();

  if (parser.parseRegion(*bodyRegion, induction, builder.getIndexType())) {
    return mlir::failure();
  }

  return mlir::success();
}

static void print(mlir::OpAsmPrinter& printer, ForEquationOp op)
{
  printer << op.getOperationName() << " " << op.induction() << " = " << op.from() << " to " << op.to();
  printer.printOptionalAttrDict(op->getAttrs(), {"from", "to"});
  printer.printRegion(op.bodyRegion(), false);
}

//===----------------------------------------------------------------------===//
// EquationOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseEquationOp(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
    return mlir::failure();
  }

  mlir::Region* bodyRegion = result.addRegion();

  if (parser.parseRegion(*bodyRegion)) {
    return mlir::failure();
  }

  return mlir::success();
}

static void print(mlir::OpAsmPrinter& printer, EquationOp op)
{
  printer << op.getOperationName();
  printer.printOptionalAttrDictWithKeyword(op->getAttrs());
  printer.printRegion(op.bodyRegion());
}

//===----------------------------------------------------------------------===//
// SubscriptionOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseSubscriptionOp(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
  auto loc = parser.getCurrentLocation();
  mlir::OpAsmParser::OperandType source;
  mlir::Type sourceType;
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> indices;
  llvm::SmallVector<mlir::Type, 3> indicesTypes;

  if (parser.parseOperand(source) ||
      parser.parseOperandList(indices, mlir::OpAsmParser::Delimiter::Square) ||
      parser.parseColonType(sourceType) ||
      parser.resolveOperand(source, sourceType, result.operands)) {
    return mlir::failure();
  }

  indicesTypes.resize(indices.size(),  mlir::IndexType::get(result.getContext()));
  size_t i = 0;

  while (mlir::succeeded(parser.parseOptionalComma())) {
    if (parser.parseType(indicesTypes[i++])) {
      return mlir::failure();
    }
  }

  if (parser.resolveOperands(indices, indicesTypes, loc, result.operands)) {
    return mlir::failure();
  }

  result.addTypes(sourceType.cast<ArrayType>().slice(indices.size()));
  return mlir::success();
}

static void print(mlir::OpAsmPrinter& printer, SubscriptionOp op)
{
  printer << op.getOperationName();
  printer << op.source() << "[" << op.indices() << "]";
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : " << op.getResult().getType();
}

//===----------------------------------------------------------------------===//
// EquationSideOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseEquationSideOp(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 1> values;
  mlir::Type resultType;
  auto loc = parser.getCurrentLocation();

  if (parser.parseOperandList(values) ||
      parser.parseColon() ||
      parser.parseType(resultType)) {
    return mlir::failure();
  }

  assert(resultType.isa<mlir::TupleType>());
  auto tupleType = resultType.cast<mlir::TupleType>();

  llvm::SmallVector<mlir::Type, 1> types(tupleType.begin(), tupleType.end());
  assert(types.size() == values.size());

  if (parser.resolveOperands(values, types, loc, result.operands)) {
    return mlir::failure();
  }

  result.addTypes(resultType);
  return mlir::success();
}

static void print(mlir::OpAsmPrinter& printer, EquationSideOp op)
{
  printer << op.getOperationName() << " ";
  printer.printOptionalAttrDict(op->getAttrs());
  printer << op.values() << " : " << op.getResult().getType();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseIfOp(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
  mlir::OpAsmParser::OperandType condition;
  mlir::Type conditionType;

  if (parser.parseLParen() ||
      parser.parseOperand(condition) ||
      parser.parseColonType(conditionType) ||
      parser.parseRParen() ||
      parser.resolveOperand(condition, conditionType, result.operands)) {
    return mlir::failure();
  }

  mlir::Region* thenRegion = result.addRegion();

  if (parser.parseRegion(*thenRegion)) {
    return mlir::failure();
  }

  mlir::Region* elseRegion = result.addRegion();

  if (mlir::succeeded(parser.parseOptionalKeyword("else"))) {
    if (parser.parseRegion(*elseRegion)) {
      return mlir::failure();
    }
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
    return mlir::failure();
  }

  return mlir::success();
}

static void print(mlir::OpAsmPrinter& printer, IfOp op)
{
  printer << op.getOperationName();
  printer << " (" << op.condition() << " : " << op.condition().getType() << ")";

  printer.printRegion(op.thenRegion());

  if (!op.elseRegion().empty()) {
    printer << " else";
    printer.printRegion(op.elseRegion());
  }

  printer.printOptionalAttrDictWithKeyword(op->getAttrs());
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseForOp(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
  mlir::Region* conditionRegion = result.addRegion();

  if (mlir::succeeded(parser.parseOptionalLParen())) {
    if (mlir::failed(parser.parseOptionalRParen())) {
      do {
        mlir::OpAsmParser::OperandType arg;
        mlir::Type argType;

        if (parser.parseOperand(arg) ||
            parser.parseColonType(argType) ||
            parser.resolveOperand(arg, argType, result.operands))
          return mlir::failure();
      } while (mlir::succeeded(parser.parseOptionalComma()));
    }

    if (parser.parseRParen()) {
      return mlir::failure();
    }
  }

  if (parser.parseKeyword("condition")) {
    return mlir::failure();
  }

  if (parser.parseRegion(*conditionRegion)) {
    return mlir::failure();
  }

  if (parser.parseKeyword("body")) {
    return mlir::failure();
  }

  mlir::Region* bodyRegion = result.addRegion();

  if (parser.parseRegion(*bodyRegion)) {
    return mlir::failure();
  }

  if (parser.parseKeyword("step")) {
    return mlir::failure();
  }

  mlir::Region* stepRegion = result.addRegion();

  if (parser.parseRegion(*stepRegion)) {
    return mlir::failure();
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
    return mlir::failure();
  }

  return mlir::success();
}

static void print(mlir::OpAsmPrinter& printer, ForOp op)
{
  printer << op.getOperationName();

  if (auto values = op.args(); !values.empty()) {
    printer << " (";

    for (auto arg : llvm::enumerate(values)) {
      if (arg.index() != 0) {
        printer << ", ";
      }

      printer << arg.value() << " : " << arg.value().getType();
    }

    printer << ")";
  }

  printer << " condition";
  printer.printRegion(op.conditionRegion(), true);
  printer << " body";
  printer.printRegion(op.bodyRegion(), true);
  printer << " step";
  printer.printRegion(op.stepRegion(), true);
  printer.printOptionalAttrDictWithKeyword(op->getAttrs());
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseWhileOp(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
  mlir::Region* conditionRegion = result.addRegion();
  mlir::Region* bodyRegion = result.addRegion();

  if (parser.parseRegion(*conditionRegion) ||
      parser.parseKeyword("do") ||
      parser.parseRegion(*bodyRegion)) {
    return mlir::failure();
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
    return mlir::failure();
  }

  if (conditionRegion->empty()) {
    conditionRegion->emplaceBlock();
  }

  if (bodyRegion->empty()) {
    bodyRegion->emplaceBlock();
  }

  return mlir::success();
}

static void print(mlir::OpAsmPrinter& printer, WhileOp op)
{
  printer << op.getOperationName();
  printer.printRegion(op.conditionRegion(), false);
  printer << " do";
  printer.printRegion(op.bodyRegion(), false);
  printer.printOptionalAttrDictWithKeyword(op->getAttrs());
}

//===----------------------------------------------------------------------===//
// MaxOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseMaxOp(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
  mlir::OpAsmParser::OperandType first;
  mlir::Type firstType;

  mlir::OpAsmParser::OperandType second;
  mlir::Type secondType;

  size_t numOperands = 1;

  if (parser.parseOperand(first)) {
    return mlir::failure();
  }

  if (mlir::succeeded(parser.parseOptionalComma())) {
    numOperands = 2;

    if (parser.parseOperand(second)) {
      return mlir::failure();
    }
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return mlir::failure();
  }

  if (parser.parseColon()) {
    return mlir::failure();
  }

  if (numOperands == 1) {
    if (parser.parseType(firstType) ||
        parser.resolveOperand(first, firstType, result.operands)) {
      return mlir::failure();
    }
  } else {
    if (parser.parseLParen() ||
        parser.parseType(firstType) ||
        parser.resolveOperand(first, firstType, result.operands) ||
        parser.parseComma() ||
        parser.parseType(secondType) ||
        parser.resolveOperand(second, secondType, result.operands) ||
        parser.parseRParen()) {
      return mlir::failure();
    }
  }

  mlir::Type resultType;

  if (parser.parseArrow() ||
      parser.parseType(resultType)) {
    return mlir::failure();
  }

  result.addTypes(resultType);

  return mlir::success();
}

static void print(mlir::OpAsmPrinter& printer, MaxOp op)
{
  printer << op.getOperationName();
  printer << " " << op.first();

  if (op->getNumOperands() == 2) {
    printer << ", " << op.second();
  }

  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  if (op->getNumOperands() == 1) {
    printer << op.first().getType();
  } else {
    printer << "(" << op.first().getType() << ", " << op.second().getType() << ")";
  }

  printer << " -> " << op.getResult().getType();
}

//===----------------------------------------------------------------------===//
// MinOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseMinOp(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
  mlir::OpAsmParser::OperandType first;
  mlir::Type firstType;

  mlir::OpAsmParser::OperandType second;
  mlir::Type secondType;

  size_t numOperands = 1;

  if (parser.parseOperand(first)) {
    return mlir::failure();
  }

  if (mlir::succeeded(parser.parseOptionalComma())) {
    numOperands = 2;

    if (parser.parseOperand(second)) {
      return mlir::failure();
    }
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return mlir::failure();
  }

  if (parser.parseColon()) {
    return mlir::failure();
  }

  if (numOperands == 1) {
    if (parser.parseType(firstType) ||
        parser.resolveOperand(first, firstType, result.operands)) {
      return mlir::failure();
    }
  } else {
    if (parser.parseLParen() ||
        parser.parseType(firstType) ||
        parser.resolveOperand(first, firstType, result.operands) ||
        parser.parseComma() ||
        parser.parseType(secondType) ||
        parser.resolveOperand(second, secondType, result.operands) ||
        parser.parseRParen()) {
      return mlir::failure();
    }
  }

  mlir::Type resultType;

  if (parser.parseArrow() ||
      parser.parseType(resultType)) {
    return mlir::failure();
  }

  result.addTypes(resultType);

  return mlir::success();
}

static void print(mlir::OpAsmPrinter& printer, MinOp op)
{
  printer << op.getOperationName();
  printer << " " << op.first();

  if (op->getNumOperands() == 2) {
    printer << ", " << op.second();
  }

  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  if (op->getNumOperands() == 1) {
    printer << op.first().getType();
  } else {
    printer << "(" << op.first().getType() << ", " << op.second().getType() << ")";
  }

  printer << " -> " << op.getResult().getType();
}

//===----------------------------------------------------------------------===//
// SizeOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseSizeOp(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
  mlir::OpAsmParser::OperandType array;
  mlir::Type arrayType;

  mlir::OpAsmParser::OperandType dimension;
  mlir::Type dimensionType;

  size_t numOperands = 1;

  if (parser.parseOperand(array)) {
    return mlir::failure();
  }

  if (mlir::succeeded(parser.parseOptionalComma())) {
    numOperands = 2;

    if (parser.parseOperand(dimension)) {
      return mlir::failure();
    }
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return mlir::failure();
  }

  if (parser.parseColon()) {
    return mlir::failure();
  }

  if (numOperands == 1) {
    if (parser.parseType(arrayType) ||
        parser.resolveOperand(array, arrayType, result.operands)) {
      return mlir::failure();
    }
  } else {
    if (parser.parseLParen() ||
        parser.parseType(arrayType) ||
        parser.resolveOperand(array, arrayType, result.operands) ||
        parser.parseComma() ||
        parser.parseType(dimensionType) ||
        parser.resolveOperand(dimension, dimensionType, result.operands) ||
        parser.parseRParen()) {
      return mlir::failure();
    }
  }

  mlir::Type resultType;

  if (parser.parseArrow() ||
      parser.parseType(resultType)) {
    return mlir::failure();
  }

  result.addTypes(resultType);

  return mlir::success();
}

static void print(mlir::OpAsmPrinter& printer, SizeOp op)
{
  printer << op.getOperationName();
  printer << " " << op.array();

  if (op->getNumOperands() == 2) {
    printer << ", " << op.dimension();
  }

  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  if (op->getNumOperands() == 1) {
    printer << op.array().getType();
  } else {
    printer << "(" << op.array().getType() << ", " << op.dimension().getType() << ")";
  }

  printer << " -> " << op.getResult().getType();
}

#define GET_OP_CLASSES
#include "marco/Dialect/Modelica/Modelica.cpp.inc"

namespace mlir::modelica
{
  //===----------------------------------------------------------------------===//
  // ModelOp
  //===----------------------------------------------------------------------===//

  SmallVector<StringRef> ModelOp::variableNames()
  {
    SmallVector<StringRef> result;

    walk<WalkOrder::PreOrder>([&](MemberCreateOp op) {
      result.push_back(op.name());
    });

    return result;
  }

  mlir::Block* ModelOp::bodyBlock()
  {
    assert(!bodyRegion().empty());
    return &bodyRegion().front();
  }

  //===----------------------------------------------------------------------===//
  // FunctionOp
  //===----------------------------------------------------------------------===//

  mlir::Block* FunctionOp::bodyBlock()
  {
    assert(body().getBlocks().size() == 1);
    return &body().front();
  }

  SmallVector<StringRef> FunctionOp::inputMemberNames()
  {
    SmallVector<StringRef> result;

    walk<WalkOrder::PreOrder>([&](MemberCreateOp op) {
      if (op.isInput()) {
        result.push_back(op.name());
      }
    });

    return result;
  }

  SmallVector<StringRef> FunctionOp::outputMemberNames()
  {
    SmallVector<StringRef> result;

    walk<WalkOrder::PreOrder>([&](MemberCreateOp op) {
      if (op.isOutput()) {
        result.push_back(op.name());
      }
    });

    return result;
  }

  SmallVector<StringRef> FunctionOp::protectedMemberNames()
  {
    SmallVector<StringRef> result;

    walk<WalkOrder::PreOrder>([&](MemberCreateOp op) {
      if (!op.isInput() && !op.isOutput()) {
        result.push_back(op.name());
      }
    });

    return result;
  }

  FunctionType FunctionOp::getType()
  {
    return getOperation()->getAttrOfType<mlir::TypeAttr>(typeAttrName()).getValue().cast<mlir::FunctionType>();
  }

  bool FunctionOp::shouldBeInlined()
  {
    if (!getOperation()->hasAttrOfType<mlir::BoolAttr>("inline")) {
      return false;
    }

    auto inlineAttribute = getOperation()->getAttrOfType<mlir::BoolAttr>("inline");
    return inlineAttribute.getValue();
  }

  std::vector<mlir::Value> FunctionOp::getMembers()
  {
    std::vector<mlir::Value> result;

    walk<WalkOrder::PreOrder>([&](MemberCreateOp op) {
      result.push_back(op.getResult());
    });

    return result;
  }

  //===----------------------------------------------------------------------===//
  // EquationOp
  //===----------------------------------------------------------------------===//

  mlir::Block* EquationOp::bodyBlock()
  {
    assert(bodyRegion().getBlocks().size() == 1);
    return &bodyRegion().front();
  }

  //===----------------------------------------------------------------------===//
  // ForEquationOp
  //===----------------------------------------------------------------------===//

  void ForEquationOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, long from, long to)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    state.addAttribute(fromAttrName(state.name), builder.getIndexAttr(from));
    state.addAttribute(toAttrName(state.name), builder.getIndexAttr(to));

    mlir::Region* bodyRegion = state.addRegion();
    builder.createBlock(bodyRegion, {}, builder.getIndexType());
  }

  mlir::Block* ForEquationOp::bodyBlock()
  {
    assert(bodyRegion().getBlocks().size() == 1);
    return &bodyRegion().front();
  }

  mlir::Value ForEquationOp::induction()
  {
    assert(bodyRegion().getNumArguments() != 0);
    return bodyRegion().getArgument(0);
  }

  //===----------------------------------------------------------------------===//
  // ModelOp
  //===----------------------------------------------------------------------===//

  mlir::RegionKind ModelOp::getRegionKind(unsigned index)
  {
    if (index == 0) {
      return mlir::RegionKind::SSACFG;
    }

    return mlir::RegionKind::Graph;
  }

  //===----------------------------------------------------------------------===//
  // ConstantOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult ConstantOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    return value();
  }

  mlir::ValueRange ConstantOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[const] = 0

    auto derivedOp = builder.create<ConstantOp>(getLoc(), getZeroAttr(getResult().getType()));
    return derivedOp->getResults();
  }

  void ConstantOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void ConstantOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // SqrtOp
  //===----------------------------------------------------------------------===//

  void SqrtOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (operand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange SqrtOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[sqrt(x)] = x' / sqrt(x) / 2

    auto loc = getLoc();

    mlir::Value derivedOperand = derivatives.lookup(operand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value sqrt = builder.create<SqrtOp>(loc, type, operand());
    mlir::Value numerator = builder.create<DivEWOp>(loc, type, derivedOperand, sqrt);
    mlir::Value two = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 2));
    auto derivedOp = builder.create<DivEWOp>(loc, type, numerator, two);

    return derivedOp->getResults();
  }

  void SqrtOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void SqrtOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // MemberLoadOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange MemberLoadOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto derivedOp = builder.create<MemberLoadOp>(getLoc(), getResult().getType(), derivatives.lookup(member()));
    return derivedOp->getResults();
  }

  void MemberLoadOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(member());
  }

  void MemberLoadOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // MemberStoreOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange MemberStoreOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // Store operations should be derived only if they store a value into
    // a member whose derivative is created by the current function. Otherwise,
    // we would create a double store into that derived member.

    assert(derivatives.contains(member()) && "Derived member not found");
    mlir::Value derivedMember = derivatives.lookup(member());
    builder.create<MemberStoreOp>(getLoc(), derivedMember, derivatives.lookup(value()));

    return llvm::None;
  }

  void MemberStoreOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(value());
    toBeDerived.push_back(member());
  }

  void MemberStoreOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // CastOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange CastOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto derivedOp = builder.create<CastOp>(getLoc(), getResult().getType(), derivatives.lookup(value()));
    return derivedOp->getResults();
  }

  void CastOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(value());
  }

  void CastOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // CallOp
  //===----------------------------------------------------------------------===//

  mlir::LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    /*
    // Check that the callee attribute was specified.
    auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");

    if (!fnAttr)
      return emitOpError("requires a 'callee' symbol reference attribute");

    auto fn = symbolTable.lookupNearestSymbolFrom<FunctionOp>(*this, fnAttr);
    if (!fn)
      return emitOpError() << "'" << fnAttr.getValue()
                           << "' does not reference a valid function";

    // Verify that the operand and result types match the callee.
    auto fnType = fn.getType();
    if (fnType.getNumInputs() != getNumOperands())
      return emitOpError("incorrect number of operands for callee");

    for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
      if (getOperand(i).getType() != fnType.getInput(i))
        return emitOpError("operand type mismatch: expected operand type ")
            << fnType.getInput(i) << ", but provided "
            << getOperand(i).getType() << " for operand number " << i;

    if (fnType.getNumResults() != getNumResults())
      return emitOpError("incorrect number of results for callee");

    for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
      if (getResult(i).getType() != fnType.getResult(i))
        return emitOpError("result type mismatch");
        */

    return success();
  }


  void CallOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    // The callee may have no arguments and no results, but still have side
    // effects (i.e. an external function writing elsewhere). Thus we need to
    // consider the call itself as if it is has side effects and prevent the
    // CSE pass to erase it.
    effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::SideEffects::DefaultResource::get());

    for (mlir::Value result : getResults()) {
      if (auto arrayType = result.getType().dyn_cast<ArrayType>()) {
        effects.emplace_back(mlir::MemoryEffects::Allocate::get(), result, mlir::SideEffects::DefaultResource::get());
        effects.emplace_back(mlir::MemoryEffects::Write::get(), result, mlir::SideEffects::DefaultResource::get());
      }
    }
  }

  mlir::ValueRange CallOp::getArgs()
  {
    return args();
  }

  unsigned int CallOp::getArgExpectedRank(unsigned int argIndex)
  {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto function = module.lookupSymbol<FunctionOp>(callee());

    if (function == nullptr) {
      // If the function is not declare, then assume that the arguments types
      // already match its hypothetical signature.

      mlir::Type argType = getArgs()[argIndex].getType();

      if (auto arrayType = argType.dyn_cast<ArrayType>())
        return arrayType.getRank();

      return 0;
    }

    mlir::Type argType = function.getArgumentTypes()[argIndex];

    if (auto arrayType = argType.dyn_cast<ArrayType>())
      return arrayType.getRank();

    return 0;
  }

  mlir::ValueRange CallOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    llvm::SmallVector<mlir::Type, 3> newResultsTypes;

    for (mlir::Type type : getResultTypes())
    {
      mlir::Type newResultType = type.cast<ArrayType>().slice(indexes.size());

      if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
        newResultType = arrayType.getElementType();

      newResultsTypes.push_back(newResultType);
    }

    llvm::SmallVector<mlir::Value, 3> newArgs;

    for (mlir::Value arg : args())
    {
      assert(arg.getType().isa<ArrayType>());
      mlir::Value newArg = builder.create<SubscriptionOp>(getLoc(), arg, indexes);

      if (auto arrayType = newArg.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
        newArg = builder.create<LoadOp>(getLoc(), newArg);

      newArgs.push_back(newArg);
    }

    auto op = builder.create<CallOp>(getLoc(), callee(), newResultsTypes, newArgs);
    return op->getResults();
  }

  mlir::LogicalResult CallOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (getNumResults() != 1) {
      return emitError("The callee must have one and only one result");
    }

    if (argumentIndex >= args().size()) {
      return emitError("Index out of bounds: " + std::to_string(argumentIndex));
    }

    if (auto size = currentResult.size(); size != 1) {
      return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");
    }

    mlir::Value toNest = currentResult[0];

    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto callee = module.lookupSymbol<FunctionOp>(this->callee());

    if (!callee->hasAttr("inverse")) {
      return emitError("Function " + callee->getName().getStringRef() + " is not invertible");
    }

    auto inverseAnnotation = callee->getAttrOfType<InverseFunctionsAttr>("inverse");

    if (!inverseAnnotation.isInvertible(argumentIndex)) {
      return emitError("Function " + callee->getName().getStringRef() + " is not invertible for argument " + std::to_string(argumentIndex));
    }

    size_t argsSize = args().size();
    llvm::SmallVector<mlir::Value, 3> args;

    for (auto arg : inverseAnnotation.getArgumentsIndexes(argumentIndex)) {
      if (arg < argsSize) {
        args.push_back(this->args()[arg]);
      } else {
        assert(arg == argsSize);
        args.push_back(toNest);
      }
    }

    auto invertedCall = builder.create<CallOp>(getLoc(), inverseAnnotation.getFunction(argumentIndex), this->args()[argumentIndex].getType(), args);

    getResult(0).replaceAllUsesWith(this->args()[argumentIndex]);
    erase();

    for (auto& use : toNest.getUses()) {
      if (use.getOwner() != invertedCall) {
        use.set(invertedCall.getResult(0));
      }
    }

    return mlir::success();
  }

  //===----------------------------------------------------------------------===//
  // Atan2Op
  //===----------------------------------------------------------------------===//

  mlir::ValueRange Atan2Op::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[atan2(y, x)] = (y' * x - y * x') / (y^2 + x^2)

    auto loc = getLoc();

    mlir::Value derivedY = derivatives.lookup(y());
    mlir::Value derivedX = derivatives.lookup(x());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value firstMul = builder.create<MulEWOp>(loc, type, derivedY, x());
    mlir::Value secondMul = builder.create<MulEWOp>(loc, type, y(), derivedX);
    mlir::Value numerator = builder.create<SubEWOp>(loc, type, firstMul, secondMul);

    mlir::Value firstSquared = builder.create<MulEWOp>(loc, type, y(), y());
    mlir::Value secondSquared = builder.create<MulEWOp>(loc, type, x(), x());
    mlir::Value denominator = builder.create<AddEWOp>(loc, type, firstSquared, secondSquared);
    auto derivedOp = builder.create<DivEWOp>(loc, type, numerator, denominator);

    return derivedOp->getResults();
  }

  void Atan2Op::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    llvm_unreachable("Not implemented");
  }

  void Atan2Op::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // AssignmentOp
  //===----------------------------------------------------------------------===//

  void AssignmentOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (value().getType().isa<ArrayType>())
      effects.emplace_back(mlir::MemoryEffects::Read::get(), value(), mlir::SideEffects::DefaultResource::get());

    if (destination().getType().isa<ArrayType>())
      effects.emplace_back(mlir::MemoryEffects::Write::get(), value(), mlir::SideEffects::DefaultResource::get());
  }

  mlir::ValueRange AssignmentOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto loc = getLoc();

    mlir::Value derivedSource = derivatives.lookup(value());
    mlir::Value derivedDestination = derivatives.lookup(destination());

    builder.create<AssignmentOp>(loc, derivedDestination, derivedSource);
    return llvm::None;
  }
  void AssignmentOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void AssignmentOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // AllocaOp
  //===----------------------------------------------------------------------===//

  void AllocaOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (auto arrayType = getResult().getType().dyn_cast<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());
    }
  }

  mlir::ValueRange AllocaOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    return builder.clone(*getOperation())->getResults();
  }

  void AllocaOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void AllocaOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // AllocOp
  //===----------------------------------------------------------------------===//

  void AllocOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (auto arrayType = getResult().getType().dyn_cast<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange AllocOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    return builder.clone(*getOperation())->getResults();
  }

  void AllocOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void AllocOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // DerFunctionOp
  //===----------------------------------------------------------------------===//

  mlir::ArrayRef<mlir::Type> DerFunctionOp::getCallableResults()
  {
    auto module = getOperation()->getParentOfType<::mlir::ModuleOp>();
    return mlir::cast<mlir::CallableOpInterface>(module.lookupSymbol(derived_function())).getCallableResults();
  }

  //===----------------------------------------------------------------------===//
  // AbsOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult AbsOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto operand = operands[0];

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        return getAttr(resultType, std::abs(getScalarIntegerLikeValue(operand)));
      }

      if (isScalarFloatLike(operand)) {
        return getAttr(resultType, std::abs(getScalarFloatLikeValue(operand)));
      }
    }

    return {};
  }

  void AbsOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (operand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange AbsOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int AbsOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange AbsOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<AbsOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  //===----------------------------------------------------------------------===//
  // AcosOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult AcosOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto operand = operands[0];

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        return getAttr(resultType, std::acos(getScalarIntegerLikeValue(operand)));
      }

      if (isScalarFloatLike(operand)) {
        return getAttr(resultType, std::acos(getScalarFloatLikeValue(operand)));
      }
    }

    return {};
  }

  void AcosOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (operand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange AcosOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int AcosOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange AcosOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<AcosOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange AcosOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[acos(x)] = -x' / sqrt(1 - x^2)

    auto loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(operand());
    mlir::Type type = convertToRealType(result().getType());

    mlir::Value one = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 1));
    mlir::Value two = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 2));
    mlir::Value argSquared = builder.create<PowEWOp>(loc, type, operand(), two);
    mlir::Value sub = builder.create<SubEWOp>(loc, type, one, argSquared);
    mlir::Value denominator = builder.create<SqrtOp>(loc, type, sub);
    mlir::Value div = builder.create<DivEWOp>(loc, type, derivedOperand, denominator);
    auto derivedOp = builder.create<NegateOp>(loc, type, div);

    return derivedOp->getResults();
  }

  void AcosOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void AcosOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // AddOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult AddOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto lhs = operands[0];
    auto rhs = operands[1];

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, getScalarIntegerLikeValue(lhs) + getScalarIntegerLikeValue(rhs));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, getScalarFloatLikeValue(lhs) + getScalarFloatLikeValue(rhs));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, getScalarIntegerLikeValue(lhs) + getScalarFloatLikeValue(rhs));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, getScalarFloatLikeValue(lhs) + getScalarIntegerLikeValue(rhs));
      }
    }

    return {};
  }

  void AddOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (lhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (rhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::LogicalResult AddOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (auto size = currentResult.size(); size != 1) {
      return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      //mlir::Type resultType = getMostGenericType(nestedOperand, get)
      auto right = builder.create<SubOp>(getLoc(), lhs().getType(), nestedOperand, rhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(lhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<SubOp>(getLoc(), rhs().getType(), nestedOperand, lhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(rhs());
      erase();

      return mlir::success();
    }

    return emitError("Can't invert the operand #" + std::to_string(argumentIndex) + ". The operation has 2 operands.");
  }

  mlir::Value AddOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(definingOp)) {
          return casted.distributeNegateOp(builder, resultType);
        }
      }

      return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<AddOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value AddOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(definingOp)) {
          return casted.distributeMulOp(builder, resultType, value);
        }
      }

      return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<AddOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value AddOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<DivOpDistributionInterface>(definingOp)) {
          return casted.distributeDivOp(builder, resultType, value);
        }
      }

      return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<AddOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange AddOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(lhs());
    mlir::Value derivedRhs = derivatives.lookup(rhs());

    auto derivedOp = builder.create<AddOp>(
        loc, convertToRealType(result().getType()), derivedLhs, derivedRhs);

    return derivedOp->getResults();
  }

  void AddOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(lhs());
    toBeDerived.push_back(rhs());
  }

  void AddOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // AddEWOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult AddEWOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto lhs = operands[0];
    auto rhs = operands[1];

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, getScalarIntegerLikeValue(lhs) + getScalarIntegerLikeValue(rhs));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, getScalarFloatLikeValue(lhs) + getScalarFloatLikeValue(rhs));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, getScalarIntegerLikeValue(lhs) + getScalarFloatLikeValue(rhs));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, getScalarFloatLikeValue(lhs) + getScalarIntegerLikeValue(rhs));
      }
    }

    return {};
  }

  void AddEWOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (lhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (rhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::LogicalResult AddEWOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (auto size = currentResult.size(); size != 1) {
      return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<SubEWOp>(getLoc(), lhs().getType(), nestedOperand, rhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(lhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<SubEWOp>(getLoc(), rhs().getType(), nestedOperand, lhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(rhs());
      erase();

      return mlir::success();
    }

    return emitError("Can't invert the operand #" + std::to_string(argumentIndex) + ". The operation has 2 operands.");
  }

  mlir::Value AddEWOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(definingOp)) {
          return casted.distributeNegateOp(builder, resultType);
        }
      }

      return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<AddEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value AddEWOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(definingOp)) {
          return casted.distributeMulOp(builder, resultType, value);
        }
      }

      return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<AddEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value AddEWOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<DivOpDistributionInterface>(definingOp)) {
          return casted.distributeDivOp(builder, resultType, value);
        }
      }

      return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<AddEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange AddEWOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(lhs());
    mlir::Value derivedRhs = derivatives.lookup(rhs());

    auto derivedOp = builder.create<AddEWOp>(
        loc, convertToRealType(result().getType()), derivedLhs, derivedRhs);

    return derivedOp->getResults();
  }

  void AddEWOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(lhs());
    toBeDerived.push_back(rhs());
  }

  void AddEWOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // AndOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult AndOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto lhs = operands[0];
    auto rhs = operands[1];

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarIntegerLikeValue(lhs) != 0 && getScalarIntegerLikeValue(rhs) != 0));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarFloatLikeValue(lhs) != 0 && getScalarFloatLikeValue(rhs) != 0));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarIntegerLikeValue(lhs) != 0 && getScalarFloatLikeValue(rhs) != 0));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarFloatLikeValue(lhs) != 0 && getScalarIntegerLikeValue(rhs) != 0));
      }
    }

    return {};
  }

  void AndOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (lhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (rhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  //===----------------------------------------------------------------------===//
  // AsinOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult AsinOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto operand = operands[0];

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        return getAttr(resultType, std::asin(getScalarIntegerLikeValue(operand)));
      }

      if (isScalarFloatLike(operand)) {
        return getAttr(resultType, std::asin(getScalarFloatLikeValue(operand)));
      }
    }

    return {};
  }

  void AsinOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (operand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange AsinOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int AsinOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange AsinOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<AsinOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange AsinOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[arcsin(x)] = x' / sqrt(1 - x^2)

    auto loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(operand());
    mlir::Type type = convertToRealType(result().getType());

    mlir::Value one = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 1));
    mlir::Value two = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 2));
    mlir::Value argSquared = builder.create<PowEWOp>(loc, type, operand(), two);
    mlir::Value sub = builder.create<SubEWOp>(loc, type, one, argSquared);
    mlir::Value denominator = builder.create<SqrtOp>(loc, type, sub);
    auto derivedOp = builder.create<DivEWOp>(loc, type, derivedOperand, denominator);

    return derivedOp->getResults();
  }

  void AsinOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void AsinOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // AtanOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult AtanOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto operand = operands[0];

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        return getAttr(resultType, std::atan(getScalarIntegerLikeValue(operand)));
      }

      if (isScalarFloatLike(operand)) {
        return getAttr(resultType, std::atan(getScalarFloatLikeValue(operand)));
      }
    }

    return {};
  }

  void AtanOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (operand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange AtanOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int AtanOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange AtanOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<AtanOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange AtanOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[atan(x)] = x' / (1 + x^2)

    auto loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(operand());
    mlir::Type type = convertToRealType(result().getType());

    mlir::Value one = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 1));
    mlir::Value two = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 2));
    mlir::Value argSquared = builder.create<PowEWOp>(loc, type, operand(), two);
    mlir::Value denominator = builder.create<AddEWOp>(loc, type, one, argSquared);
    auto derivedOp = builder.create<DivEWOp>(loc, type, derivedOperand, denominator);

    return derivedOp->getResults();
  }

  void AtanOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void AtanOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // Atan2Op
  //===----------------------------------------------------------------------===//

  void Atan2Op::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (y().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), y(), mlir::SideEffects::DefaultResource::get());
    }

    if (x().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), x(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange Atan2Op::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int Atan2Op::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange Atan2Op::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newY = builder.create<SubscriptionOp>(getLoc(), y(), indexes);

    if (auto arrayType = newY.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newY = builder.create<LoadOp>(getLoc(), newY);
    }

    mlir::Value newX = builder.create<SubscriptionOp>(getLoc(), x(), indexes);

    if (auto arrayType = newX.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newX = builder.create<LoadOp>(getLoc(), newX);
    }

    auto op = builder.create<Atan2Op>(getLoc(), newResultType, newY, newX);
    return op->getResults();
  }

  //===----------------------------------------------------------------------===//
  // CosOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult CosOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto operand = operands[0];

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        return getAttr(resultType, std::cos(getScalarIntegerLikeValue(operand)));
      }

      if (isScalarFloatLike(operand)) {
        return getAttr(resultType, std::cos(getScalarFloatLikeValue(operand)));
      }
    }

    return {};
  }

  void CosOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (operand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange CosOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int CosOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange CosOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<CosOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange CosOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[cos(x)] = -x' * sin(x)

    auto loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(operand());
    mlir::Type type = convertToRealType(result().getType());

    mlir::Value sin = builder.create<SinOp>(loc, type, operand());
    mlir::Value negatedSin = builder.create<NegateOp>(loc, type, sin);
    auto derivedOp = builder.create<MulEWOp>(loc, type, negatedSin, derivedOperand);

    return derivedOp->getResults();
  }

  void CosOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void CosOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // CoshOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult CoshOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto operand = operands[0];

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        return getAttr(resultType, std::cosh(getScalarIntegerLikeValue(operand)));
      }

      if (isScalarFloatLike(operand)) {
        return getAttr(resultType, std::cosh(getScalarFloatLikeValue(operand)));
      }
    }

    return {};
  }

  void CoshOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (operand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange CoshOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int CoshOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange CoshOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<CoshOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange CoshOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[cosh(x)] = x' * sinh(x)

    auto loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(operand());
    mlir::Type type = convertToRealType(result().getType());

    mlir::Value sinh = builder.create<SinhOp>(loc, type, operand());
    auto derivedOp = builder.create<MulEWOp>(loc, type, sinh, derivedOperand);

    return derivedOp->getResults();
  }

  void CoshOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void CoshOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // DiagonalOp
  //===----------------------------------------------------------------------===//

  void DiagonalOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), values(), mlir::SideEffects::DefaultResource::get());
    effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // DivOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult DivOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto lhs = operands[0];
    auto rhs = operands[1];

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, getScalarIntegerLikeValue(lhs) / getScalarIntegerLikeValue(rhs));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, getScalarFloatLikeValue(lhs) / getScalarFloatLikeValue(rhs));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, getScalarIntegerLikeValue(lhs) / getScalarFloatLikeValue(rhs));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, getScalarFloatLikeValue(lhs) / getScalarIntegerLikeValue(rhs));
      }
    }

    return {};
  }

  void DivOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (lhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (rhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::LogicalResult DivOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (auto size = currentResult.size(); size != 1) {
      return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<MulOp>(getLoc(), lhs().getType(), nestedOperand, rhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      getResult().replaceAllUsesWith(lhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<DivOp>(getLoc(), rhs().getType(), lhs(), nestedOperand);

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      getResult().replaceAllUsesWith(rhs());
      erase();

      return mlir::success();
    }

    return emitError("Index out of bounds: " + std::to_string(argumentIndex));
  }

  mlir::Value DivOp::distribute(mlir::OpBuilder& builder)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(*this);

    auto lhsDefiningOp = lhs().getDefiningOp();

    if (!lhsDefiningOp) {
      return getResult();
    }

    if (!mlir::isa<DivOpDistributionInterface>(lhsDefiningOp)) {
      // The operation can't be propagated because the dividend does not
      // know how to distribute the division to their children.
      return getResult();
    }

    DivOpDistributionInterface childOp = mlir::cast<DivOpDistributionInterface>(lhsDefiningOp);
    mlir::Value toDistribute = rhs();

    return childOp.distributeDivOp(builder, result().getType(), toDistribute);
  }

  mlir::Value DivOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(definingOp)) {
          return casted.distributeNegateOp(builder, resultType);
        }
      }

      return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<DivOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value DivOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(definingOp)) {
          return casted.distributeMulOp(builder, resultType, value);
        }
      }

      return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<DivOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value DivOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<DivOpDistributionInterface>(definingOp)) {
          return casted.distributeDivOp(builder, resultType, value);
        }
      }

      return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<DivOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange DivOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(lhs());
    mlir::Value derivedRhs = derivatives.lookup(rhs());

    mlir::Type type = convertToRealType(result().getType());

    mlir::Value firstMul = builder.create<MulOp>(loc, type, derivedLhs, rhs());
    mlir::Value secondMul = builder.create<MulOp>(loc, type, lhs(), derivedRhs);
    mlir::Value numerator = builder.create<SubOp>(loc, type, firstMul, secondMul);
    mlir::Value two = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 2));
    mlir::Value denominator = builder.create<PowOp>(loc, convertToRealType(rhs().getType()), rhs(), two);
    auto derivedOp = builder.create<DivOp>(loc, type, numerator, denominator);

    return derivedOp->getResults();
  }

  void DivOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(lhs());
    toBeDerived.push_back(rhs());
  }

  void DivOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // DivEWOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult DivEWOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto lhs = operands[0];
    auto rhs = operands[1];

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, getScalarIntegerLikeValue(lhs) / getScalarIntegerLikeValue(rhs));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, getScalarFloatLikeValue(lhs) / getScalarFloatLikeValue(rhs));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, getScalarIntegerLikeValue(lhs) / getScalarFloatLikeValue(rhs));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, getScalarFloatLikeValue(lhs) / getScalarIntegerLikeValue(rhs));
      }
    }

    return {};
  }

  void DivEWOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (lhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (rhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::LogicalResult DivEWOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (auto size = currentResult.size(); size != 1) {
      return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<MulEWOp>(getLoc(), lhs().getType(), nestedOperand, rhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      getResult().replaceAllUsesWith(lhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<DivEWOp>(getLoc(), rhs().getType(), lhs(), nestedOperand);

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      getResult().replaceAllUsesWith(rhs());
      erase();

      return mlir::success();
    }

    return emitError("Index out of bounds: " + std::to_string(argumentIndex));
  }

  mlir::Value DivEWOp::distribute(mlir::OpBuilder& builder)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(*this);

    auto lhsDefiningOp = lhs().getDefiningOp();

    if (!lhsDefiningOp) {
      return getResult();
    }

    if (!mlir::isa<DivOpDistributionInterface>(lhsDefiningOp)) {
      // The operation can't be propagated because the dividend does not
      // know how to distribute the division to their children.
      return getResult();
    }

    DivOpDistributionInterface childOp = mlir::cast<DivOpDistributionInterface>(lhsDefiningOp);
    mlir::Value toDistribute = rhs();

    return childOp.distributeDivOp(builder, result().getType(), toDistribute);
  }

  mlir::Value DivEWOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(definingOp)) {
          return casted.distributeNegateOp(builder, resultType);
        }
      }

      return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<DivEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value DivEWOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(definingOp)) {
          return casted.distributeMulOp(builder, resultType, value);
        }
      }

      return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<DivEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value DivEWOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<DivOpDistributionInterface>(definingOp)) {
          return casted.distributeDivOp(builder, resultType, value);
        }
      }

      return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<DivEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange DivEWOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(lhs());
    mlir::Value derivedRhs = derivatives.lookup(rhs());

    mlir::Type type = convertToRealType(result().getType());

    mlir::Value firstMul = builder.create<MulEWOp>(loc, type, derivedLhs, rhs());
    mlir::Value secondMul = builder.create<MulEWOp>(loc, type, lhs(), derivedRhs);
    mlir::Value numerator = builder.create<SubEWOp>(loc, type, firstMul, secondMul);
    mlir::Value two = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 2));
    mlir::Value denominator = builder.create<PowEWOp>(loc, convertToRealType(rhs().getType()), rhs(), two);
    auto derivedOp = builder.create<DivEWOp>(loc, type, numerator, denominator);

    return derivedOp->getResults();
  }

  void DivEWOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(lhs());
    toBeDerived.push_back(rhs());
  }

  void DivEWOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // ExpOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult ExpOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto operand = operands[0];

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        return getAttr(resultType, std::exp(getScalarIntegerLikeValue(operand)));
      }

      if (isScalarFloatLike(operand)) {
        return getAttr(resultType, std::exp(getScalarFloatLikeValue(operand)));
      }
    }

    return {};
  }

  void ExpOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (exponent().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), exponent(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange ExpOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int ExpOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange ExpOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), exponent(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<ExpOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange ExpOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[e^x] = x' * e^x

    auto loc = getLoc();
    mlir::Value derivedExponent = derivatives.lookup(exponent());
    mlir::Type type = convertToRealType(result().getType());

    mlir::Value pow = builder.create<ExpOp>(loc, type, exponent());
    auto derivedOp = builder.create<MulEWOp>(loc, type, pow, derivedExponent);

    return derivedOp->getResults();
  }

  void ExpOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(exponent());
  }

  void ExpOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // ForOp
  //===----------------------------------------------------------------------===//

  mlir::Block* ForOp::conditionBlock()
  {
    assert(!conditionRegion().empty());
    return &conditionRegion().front();
  }

  mlir::Block* ForOp::bodyBlock()
  {
    assert(!bodyRegion().empty());
    return &bodyRegion().front();
  }

  mlir::Block* ForOp::stepBlock()
  {
    assert(!stepRegion().empty());
    return &stepRegion().front();
  }

  mlir::ValueRange ForOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    return llvm::None;
  }

  void ForOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void ForOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    regions.push_back(&bodyRegion());
  }

  //===----------------------------------------------------------------------===//
  // FreeOp
  //===----------------------------------------------------------------------===//

  void FreeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Free::get(), array(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // IdentityOp
  //===----------------------------------------------------------------------===//

  void IdentityOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // IfOp
  //===----------------------------------------------------------------------===//

  void IfOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, Value condition, bool withElseRegion)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    state.addOperands(condition);

    // Create the "then" region
    mlir::Region* thenRegion = state.addRegion();
    builder.createBlock(thenRegion);

    // Create the "else" region
    mlir::Region* elseRegion = state.addRegion();

    if (withElseRegion) {
      builder.createBlock(elseRegion);
    }
  }

  mlir::Block* IfOp::thenBlock()
  {
    return &thenRegion().front();
  }

  mlir::Block* IfOp::elseBlock()
  {
    return &elseRegion().front();
  }

  mlir::ValueRange IfOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    return llvm::None;
  }

  void IfOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void IfOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    regions.push_back(&thenRegion());
    regions.push_back(&elseRegion());
  }

  //===----------------------------------------------------------------------===//
  // LinspaceOp
  //===----------------------------------------------------------------------===//

  void LinspaceOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // LoadOp
  //===----------------------------------------------------------------------===//

  void LoadOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), array(), mlir::SideEffects::DefaultResource::get());
  }

  mlir::ValueRange LoadOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto derivedOp = builder.create<LoadOp>(getLoc(), derivatives.lookup(array()), indices());
    return derivedOp->getResults();
  }

  void LoadOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(array());
  }

  void LoadOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // LogOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult LogOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto operand = operands[0];

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        return getAttr(resultType, std::log(getScalarIntegerLikeValue(operand)));
      }

      if (isScalarFloatLike(operand)) {
        return getAttr(resultType, std::log(getScalarFloatLikeValue(operand)));
      }
    }

    return {};
  }

  void LogOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (operand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange LogOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int LogOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange LogOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<LogOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange LogOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[ln(x)] = x' / x

    mlir::Value derivedOperand = derivatives.lookup(operand());

    auto derivedOp = builder.create<DivEWOp>(
        getLoc(), convertToRealType(result().getType()), derivedOperand, operand());

    return derivedOp->getResults();
  }

  void LogOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void LogOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // Log10Op
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult Log10Op::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto operand = operands[0];

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        return getAttr(resultType, std::log10(getScalarIntegerLikeValue(operand)));
      }

      if (isScalarFloatLike(operand)) {
        return getAttr(resultType, std::log10(getScalarFloatLikeValue(operand)));
      }
    }

    return {};
  }

  void Log10Op::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (operand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange Log10Op::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int Log10Op::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange Log10Op::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<Log10Op>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange Log10Op::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[log10(x)] = x' / (x * ln(10))

    auto loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(operand());
    mlir::Type type = convertToRealType(result().getType());

    mlir::Value ten = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 10));
    mlir::Value log = builder.create<LogOp>(loc, RealType::get(getContext()), ten);
    mlir::Value mul = builder.create<MulEWOp>(loc, type, operand(), log);
    auto derivedOp = builder.create<DivEWOp>(loc, result().getType(), derivedOperand, mul);

    return derivedOp->getResults();
  }

  void Log10Op::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void Log10Op::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // MaxOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult MaxOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    if (operands.size() == 2) {
      auto first = operands[0];
      auto second = operands[1];

      if (!first || !second) {
        return {};
      }

      auto resultType = getResult().getType();

      if (isScalar(first) && isScalar(second)) {
        if (isScalarIntegerLike(first) && isScalarIntegerLike(second)) {
          return getAttr(resultType, std::max(getScalarIntegerLikeValue(first), getScalarIntegerLikeValue(second)));
        }

        if (isScalarFloatLike(first) && isScalarFloatLike(second)) {
          return getAttr(resultType, std::max(getScalarFloatLikeValue(first), getScalarFloatLikeValue(second)));
        }

        if (isScalarIntegerLike(first) && isScalarFloatLike(second)) {
          auto firstValue = getScalarIntegerLikeValue(first);
          auto secondValue = getScalarFloatLikeValue(second);

          if (firstValue >= secondValue) {
            return getAttr(resultType, firstValue);
          } else {
            return getAttr(resultType, secondValue);
          }
        }

        if (isScalarFloatLike(first) && isScalarIntegerLike(second)) {
          auto firstValue = getScalarFloatLikeValue(first);
          auto secondValue = getScalarIntegerLikeValue(second);

          if (firstValue >= secondValue) {
            return getAttr(resultType, firstValue);
          } else {
            return getAttr(resultType, secondValue);
          }
        }
      }
    }

    return {};
  }

  void MaxOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (first().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), first(), mlir::SideEffects::DefaultResource::get());
    }
  }

  //===----------------------------------------------------------------------===//
  // MinOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult MinOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    if (operands.size() == 2) {
      auto first = operands[0];
      auto second = operands[1];

      if (!first || !second) {
        return {};
      }

      auto resultType = getResult().getType();

      if (isScalar(first) && isScalar(second)) {
        if (isScalarIntegerLike(first) && isScalarIntegerLike(second)) {
          return getAttr(resultType, std::min(getScalarIntegerLikeValue(first), getScalarIntegerLikeValue(second)));
        }

        if (isScalarFloatLike(first) && isScalarFloatLike(second)) {
          return getAttr(resultType, std::min(getScalarFloatLikeValue(first), getScalarFloatLikeValue(second)));
        }

        if (isScalarIntegerLike(first) && isScalarFloatLike(second)) {
          auto firstValue = getScalarIntegerLikeValue(first);
          auto secondValue = getScalarFloatLikeValue(second);

          if (firstValue <= secondValue) {
            return getAttr(resultType, firstValue);
          } else {
            return getAttr(resultType, secondValue);
          }
        }

        if (isScalarFloatLike(first) && isScalarIntegerLike(second)) {
          auto firstValue = getScalarFloatLikeValue(first);
          auto secondValue = getScalarIntegerLikeValue(second);

          if (firstValue <= secondValue) {
            return getAttr(resultType, firstValue);
          } else {
            return getAttr(resultType, secondValue);
          }
        }
      }
    }

    return {};
  }

  void MinOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (first().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), first(), mlir::SideEffects::DefaultResource::get());
    }
  }

  //===----------------------------------------------------------------------===//
  // MulOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult MulOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto lhs = operands[0];
    auto rhs = operands[1];

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, getScalarIntegerLikeValue(lhs) * getScalarIntegerLikeValue(rhs));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, getScalarFloatLikeValue(lhs) * getScalarFloatLikeValue(rhs));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, getScalarIntegerLikeValue(lhs) * getScalarFloatLikeValue(rhs));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, getScalarFloatLikeValue(lhs) * getScalarIntegerLikeValue(rhs));
      }
    }

    return {};
  }

  void MulOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (lhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (rhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::LogicalResult MulOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (auto size = currentResult.size(); size != 1) {
      return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<DivOp>(getLoc(), lhs().getType(), nestedOperand, rhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
          use.set(right.getResult());
      }

      replaceAllUsesWith(lhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<DivOp>(getLoc(), rhs().getType(), nestedOperand, lhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
          use.set(right.getResult());
      }

      getResult().replaceAllUsesWith(rhs());
      erase();

      return mlir::success();
    }

    return emitError("Index out of bounds: " + std::to_string(argumentIndex));
  }

  mlir::Value MulOp::distribute(mlir::OpBuilder& builder)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(*this);

    auto lhsDefiningOp = lhs().getDefiningOp();
    auto rhsDefiningOp = rhs().getDefiningOp();

    if (!lhsDefiningOp && !rhsDefiningOp) {
      return getResult();
    }

    if (!mlir::isa<MulOpDistributionInterface>(lhsDefiningOp) && !mlir::isa<MulOpDistributionInterface>(rhsDefiningOp)) {
      // The operation can't be propagated because none of the children
      // know how to distribute the multiplication to their children.
      return getResult();
    }

    MulOpDistributionInterface childOp;
    mlir::Value toDistribute;

    if (lhsDefiningOp != nullptr && mlir::isa<MulOpDistributionInterface>(lhsDefiningOp)) {
      childOp = mlir::cast<MulOpDistributionInterface>(lhsDefiningOp);
      toDistribute = rhs();
    } else {
      assert(rhsDefiningOp != nullptr);
      childOp = mlir::cast<MulOpDistributionInterface>(rhsDefiningOp);
      toDistribute = lhs();
    }

    assert(childOp != nullptr && toDistribute != nullptr);
    return childOp.distributeMulOp(builder, result().getType(), toDistribute);
  }

  mlir::Value MulOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(definingOp)) {
          return casted.distributeNegateOp(builder, resultType);
        }
      }

      return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<MulOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value MulOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(definingOp)) {
          return casted.distributeMulOp(builder, resultType, value);
        }
      }

      return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<MulOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value MulOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<DivOpDistributionInterface>(definingOp)) {
          return casted.distributeDivOp(builder, resultType, value);
        }
      }

      return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<MulOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange MulOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(lhs());
    mlir::Value derivedRhs = derivatives.lookup(rhs());

    mlir::Type type = convertToRealType(result().getType());

    mlir::Value firstMul = builder.create<MulOp>(loc, type, derivedLhs, rhs());
    mlir::Value secondMul = builder.create<MulOp>(loc, type, lhs(), derivedRhs);
    auto derivedOp = builder.create<AddOp>(loc, type, firstMul, secondMul);

    return derivedOp->getResults();
  }

  void MulOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(lhs());
    toBeDerived.push_back(rhs());
  }

  void MulOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // MulEWOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult MulEWOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto lhs = operands[0];
    auto rhs = operands[1];

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, getScalarIntegerLikeValue(lhs) * getScalarIntegerLikeValue(rhs));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, getScalarFloatLikeValue(lhs) * getScalarFloatLikeValue(rhs));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, getScalarIntegerLikeValue(lhs) * getScalarFloatLikeValue(rhs));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, getScalarFloatLikeValue(lhs) * getScalarIntegerLikeValue(rhs));
      }
    }

    return {};
  }

  void MulEWOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (lhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (rhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::LogicalResult MulEWOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (auto size = currentResult.size(); size != 1) {
      return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<DivEWOp>(getLoc(), lhs().getType(), nestedOperand, rhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(lhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<DivEWOp>(getLoc(), rhs().getType(), nestedOperand, lhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      getResult().replaceAllUsesWith(rhs());
      erase();

      return mlir::success();
    }

    return emitError("Index out of bounds: " + std::to_string(argumentIndex));
  }

  mlir::Value MulEWOp::distribute(mlir::OpBuilder& builder)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(*this);

    auto lhsDefiningOp = lhs().getDefiningOp();
    auto rhsDefiningOp = rhs().getDefiningOp();

    if (!lhsDefiningOp && !rhsDefiningOp) {
      return getResult();
    }

    if (!mlir::isa<MulOpDistributionInterface>(lhs().getDefiningOp()) && !mlir::isa<MulOpDistributionInterface>(rhs().getDefiningOp())) {
      // The operation can't be propagated because none of the children
      // know how to distribute the multiplication to their children.
      return getResult();
    }

    MulOpDistributionInterface childOp;
    mlir::Value toDistribute;

    if (lhsDefiningOp != nullptr && mlir::isa<MulOpDistributionInterface>(lhsDefiningOp)) {
      childOp = mlir::cast<MulOpDistributionInterface>(lhsDefiningOp);
      toDistribute = rhs();
    } else {
      assert(rhsDefiningOp != nullptr);
      childOp = mlir::cast<MulOpDistributionInterface>(rhsDefiningOp);
      toDistribute = lhs();
    }

    assert(childOp != nullptr && toDistribute != nullptr);
    return childOp.distributeMulOp(builder, result().getType(), toDistribute);
  }

  mlir::Value MulEWOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(definingOp)) {
          return casted.distributeNegateOp(builder, resultType);
        }
      }

      return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<MulEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value MulEWOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(definingOp)) {
          return casted.distributeMulOp(builder, resultType, value);
        }
      }

      return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<MulEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value MulEWOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<DivOpDistributionInterface>(definingOp)) {
          return casted.distributeDivOp(builder, resultType, value);
        }
      }

      return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<MulEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange MulEWOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(lhs());
    mlir::Value derivedRhs = derivatives.lookup(rhs());

    mlir::Type type = convertToRealType(result().getType());

    mlir::Value firstMul = builder.create<MulEWOp>(loc, type, derivedLhs, rhs());
    mlir::Value secondMul = builder.create<MulEWOp>(loc, type, lhs(), derivedRhs);
    auto derivedOp = builder.create<AddEWOp>(loc, type, firstMul, secondMul);

    return derivedOp->getResults();
  }

  void MulEWOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(lhs());
    toBeDerived.push_back(rhs());
  }

  void MulEWOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // NegateOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult NegateOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto operand = operands[0];

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        return getAttr(resultType, -1 * getScalarIntegerLikeValue(operand));
      }

      if (isScalarFloatLike(operand)) {
        return getAttr(resultType, -1 * getScalarFloatLikeValue(operand));
      }
    }

    return {};
  }

  void NegateOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (operand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::LogicalResult NegateOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (argumentIndex > 0) {
      return emitError("Index out of bounds: " + std::to_string(argumentIndex));
    }

    if (auto size = currentResult.size(); size != 1) {
      return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");
    }

    mlir::Value toNest = currentResult[0];

    mlir::Value nestedOperand = readValue(builder, toNest);
    auto right = builder.create<NegateOp>(getLoc(), operand().getType(), nestedOperand);

    for (auto& use : toNest.getUses()) {
      if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
        use.set(right.getResult());
      }
    }

    replaceAllUsesWith(operand());
    erase();

    return mlir::success();
  }

  mlir::Value NegateOp::distribute(mlir::OpBuilder& builder)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(*this);

    auto operandDefiningOp = operand().getDefiningOp();

    if (!operandDefiningOp) {
      return getResult();
    }

    if (auto childOp = mlir::dyn_cast<NegateOpDistributionInterface>(operand().getDefiningOp())) {
      return childOp.distributeNegateOp(builder, result().getType());
    }

    // The operation can't be propagated because the child doesn't
    // know how to distribute the multiplication to its children.
    return getResult();
  }

  mlir::Value NegateOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(definingOp)) {
          return casted.distributeNegateOp(builder, resultType);
        }
      }

      return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
    };

    mlir::Value operand = distributeFn(this->operand());

    return builder.create<NegateOp>(getLoc(), resultType, operand);
  }

  mlir::Value NegateOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(definingOp)) {
          return casted.distributeMulOp(builder, resultType, value);
        }
      }

      return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value operand = distributeFn(this->operand());

    return builder.create<NegateOp>(getLoc(), resultType, operand);
  }

  mlir::Value NegateOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<DivOpDistributionInterface>(definingOp)) {
          return casted.distributeDivOp(builder, resultType, value);
        }
      }

      return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value operand = distributeFn(this->operand());

    return builder.create<NegateOp>(getLoc(), resultType, operand);
  }

  mlir::ValueRange NegateOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    mlir::Value derivedOperand = derivatives.lookup(operand());
    auto derivedOp = builder.create<NegateOp>(getLoc(), convertToRealType(result().getType()), derivedOperand);
    return derivedOp->getResults();
  }

  void NegateOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void NegateOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // NotOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult NotOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto operand = operands[0];

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        return getAttr(resultType, static_cast<long>(getScalarIntegerLikeValue(operand) == 0));
      }

      if (isScalarFloatLike(operand)) {
        return getAttr(resultType, static_cast<long>(getScalarFloatLikeValue(operand) == 0));
      }
    }

    return {};
  }

  void NotOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (operand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  //===----------------------------------------------------------------------===//
  // OnesOp
  //===----------------------------------------------------------------------===//

  void OnesOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // OrOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult OrOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto lhs = operands[0];
    auto rhs = operands[1];

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarIntegerLikeValue(lhs) != 0 || getScalarIntegerLikeValue(rhs) != 0));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarFloatLikeValue(lhs) != 0 || getScalarFloatLikeValue(rhs) != 0));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarIntegerLikeValue(lhs) != 0 || getScalarFloatLikeValue(rhs) != 0));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarFloatLikeValue(lhs) != 0 || getScalarIntegerLikeValue(rhs) != 0));
      }
    }

    return {};
  }

  void OrOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (lhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (rhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  //===----------------------------------------------------------------------===//
  // PowOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult PowOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto base = operands[0];
    auto exponent = operands[1];

    if (!base || !exponent) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(base) && isScalar(exponent)) {
      if (isScalarIntegerLike(base) && isScalarIntegerLike(exponent)) {
        return getAttr(resultType, std::pow(getScalarIntegerLikeValue(base), getScalarIntegerLikeValue(exponent)));
      }

      if (isScalarFloatLike(base) && isScalarFloatLike(exponent)) {
        return getAttr(resultType, std::pow(getScalarFloatLikeValue(base), getScalarFloatLikeValue(exponent)));
      }

      if (isScalarIntegerLike(base) && isScalarFloatLike(exponent)) {
        return getAttr(resultType, std::pow(getScalarIntegerLikeValue(base), getScalarFloatLikeValue(exponent)));
      }

      if (isScalarFloatLike(base) && isScalarIntegerLike(exponent)) {
        return getAttr(resultType, std::pow(getScalarFloatLikeValue(base), getScalarIntegerLikeValue(exponent)));
      }
    }

    return {};
  }

  void PowOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (base().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), base(), mlir::SideEffects::DefaultResource::get());
    }

    if (exponent().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), exponent(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange PowOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[x ^ y] = (x ^ y) * (y' * ln(x) + (y * x') / x)

    auto loc = getLoc();

    mlir::Value derivedBase = derivatives.lookup(base());
    mlir::Value derivedExponent = derivatives.lookup(exponent());

    mlir::Type type = convertToRealType(result().getType());

    mlir::Value pow = builder.create<PowOp>(loc, type, base(), exponent());
    mlir::Value ln = builder.create<LogOp>(loc, type, base());
    mlir::Value firstOperand = builder.create<MulOp>(loc, type, derivedExponent, ln);
    mlir::Value numerator = builder.create<MulOp>(loc, type, exponent(), derivedBase);
    mlir::Value secondOperand = builder.create<DivOp>(loc, type, numerator, base());
    mlir::Value sum = builder.create<AddOp>(loc, type, firstOperand, secondOperand);
    auto derivedOp = builder.create<MulOp>(loc, type, pow, sum);

    return derivedOp->getResults();
  }

  void PowOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(base());
    toBeDerived.push_back(exponent());
  }

  void PowOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // PowEWOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult PowEWOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto base = operands[0];
    auto exponent = operands[1];

    if (!base || !exponent) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(base) && isScalar(exponent)) {
      if (isScalarIntegerLike(base) && isScalarIntegerLike(exponent)) {
        return getAttr(resultType, std::pow(getScalarIntegerLikeValue(base), getScalarIntegerLikeValue(exponent)));
      }

      if (isScalarFloatLike(base) && isScalarFloatLike(exponent)) {
        return getAttr(resultType, std::pow(getScalarFloatLikeValue(base), getScalarFloatLikeValue(exponent)));
      }

      if (isScalarIntegerLike(base) && isScalarFloatLike(exponent)) {
        return getAttr(resultType, std::pow(getScalarIntegerLikeValue(base), getScalarFloatLikeValue(exponent)));
      }

      if (isScalarFloatLike(base) && isScalarIntegerLike(exponent)) {
        return getAttr(resultType, std::pow(getScalarFloatLikeValue(base), getScalarIntegerLikeValue(exponent)));
      }
    }

    return {};
  }

  void PowEWOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (base().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), base(), mlir::SideEffects::DefaultResource::get());
    }

    if (exponent().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), exponent(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange PowEWOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[x ^ y] = (x ^ y) * (y' * ln(x) + (y * x') / x)

    auto loc = getLoc();

    mlir::Value derivedBase = derivatives.lookup(base());
    mlir::Value derivedExponent = derivatives.lookup(exponent());

    mlir::Type type = convertToRealType(result().getType());

    mlir::Value pow = builder.create<PowEWOp>(loc, type, base(), exponent());
    mlir::Value ln = builder.create<LogOp>(loc, type, base());
    mlir::Value firstOperand = builder.create<MulEWOp>(loc, type, derivedExponent, ln);
    mlir::Value numerator = builder.create<MulEWOp>(loc, type, exponent(), derivedBase);
    mlir::Value secondOperand = builder.create<DivEWOp>(loc, type, numerator, base());
    mlir::Value sum = builder.create<AddEWOp>(loc, type, firstOperand, secondOperand);
    auto derivedOp = builder.create<MulEWOp>(loc, type, pow, sum);

    return derivedOp->getResults();
  }

  void PowEWOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(base());
    toBeDerived.push_back(exponent());
  }

  void PowEWOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // SignOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult SignOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto operand = operands[0];

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        auto value = getScalarIntegerLikeValue(operand);

        if (value == 0) {
          return getAttr(resultType, 0l);
        } else if (value > 0) {
          return getAttr(resultType, 1l);
        } else {
          return getAttr(resultType, -1l);
        }
      }

      if (isScalarFloatLike(operand)) {
        auto value = getScalarFloatLikeValue(operand);

        if (value == 0) {
          return getAttr(resultType, 0l);
        } else if (value > 0) {
          return getAttr(resultType, 1l);
        } else {
          return getAttr(resultType, -1l);
        }
      }
    }

    return {};
  }

  void SignOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (operand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange SignOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int SignOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange SignOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
      newResultType = arrayType.getElementType();

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);

    auto op = builder.create<SignOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  //===----------------------------------------------------------------------===//
  // SinOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult SinOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto operand = operands[0];

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        return getAttr(resultType, std::sin(getScalarIntegerLikeValue(operand)));
      }

      if (isScalarFloatLike(operand)) {
        return getAttr(resultType, std::sin(getScalarFloatLikeValue(operand)));
      }
    }

    return {};
  }

  void SinOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (operand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange SinOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int SinOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange SinOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
      newResultType = arrayType.getElementType();

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);

    auto op = builder.create<SinOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange SinOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[sin(x)] = x' * cos(x)

    auto loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(operand());
    mlir::Type type = convertToRealType(result().getType());

    mlir::Value cos = builder.create<CosOp>(loc, type, operand());
    auto derivedOp = builder.create<MulEWOp>(loc, type, cos, derivedOperand);

    return derivedOp->getResults();
  }

  void SinOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void SinOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // SinhOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult SinhOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto operand = operands[0];

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        return getAttr(resultType, std::sinh(getScalarIntegerLikeValue(operand)));
      }

      if (isScalarFloatLike(operand)) {
        return getAttr(resultType, std::sinh(getScalarFloatLikeValue(operand)));
      }
    }

    return {};
  }

  void SinhOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (operand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange SinhOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int SinhOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange SinhOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<SinhOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange SinhOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[sinh(x)] = x' * cosh(x)

    auto loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(operand());
    mlir::Type type = convertToRealType(result().getType());

    mlir::Value cosh = builder.create<CoshOp>(loc, type, operand());
    auto derivedOp = builder.create<MulEWOp>(loc, type, cosh, derivedOperand);

    return derivedOp->getResults();
  }

  void SinhOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void SinhOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // SizeOp
  //===----------------------------------------------------------------------===//

  void SizeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), array(), mlir::SideEffects::DefaultResource::get());

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  //===----------------------------------------------------------------------===//
  // SqrtOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult SqrtOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto operand = operands[0];

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        return getAttr(resultType, std::sqrt(getScalarIntegerLikeValue(operand)));
      }

      if (isScalarFloatLike(operand)) {
        return getAttr(resultType, std::sqrt(getScalarFloatLikeValue(operand)));
      }
    }

    return {};
  }

  mlir::ValueRange SqrtOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int SqrtOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange SqrtOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<SqrtOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  //===----------------------------------------------------------------------===//
  // StoreOp
  //===----------------------------------------------------------------------===//

  void StoreOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Write::get(), array(), mlir::SideEffects::DefaultResource::get());
  }

  mlir::ValueRange StoreOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto derivedOp = builder.create<StoreOp>(
        getLoc(), derivatives.lookup(value()), derivatives.lookup(array()), indices());

    return derivedOp->getResults();
  }

  void StoreOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(array());
    toBeDerived.push_back(value());
  }

  void StoreOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // SubOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult SubOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto lhs = operands[0];
    auto rhs = operands[1];

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, getScalarIntegerLikeValue(lhs) - getScalarIntegerLikeValue(rhs));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, getScalarFloatLikeValue(lhs) - getScalarFloatLikeValue(rhs));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, getScalarIntegerLikeValue(lhs) - getScalarFloatLikeValue(rhs));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, getScalarFloatLikeValue(lhs) - getScalarIntegerLikeValue(rhs));
      }
    }

    return {};
  }

  void SubOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (lhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (rhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::LogicalResult SubOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (auto size = currentResult.size(); size != 1) {
      return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<AddOp>(getLoc(), lhs().getType(), nestedOperand, rhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(lhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<SubOp>(getLoc(), rhs().getType(), lhs(), nestedOperand);

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(rhs());
      erase();

      return mlir::success();
    }

    return emitError("Can't invert the operand #" + std::to_string(argumentIndex) + ". The operation has 2 operands.");
  }

  mlir::Value SubOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(definingOp)) {
          return casted.distributeNegateOp(builder, resultType);
        }
      }

      return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<SubOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value SubOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(definingOp)) {
          return casted.distributeMulOp(builder, resultType, value);
        }
      }

      return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<SubOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value SubOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<DivOpDistributionInterface>(definingOp)) {
          return casted.distributeDivOp(builder, resultType, value);
        }
      }

      return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<SubOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange SubOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(lhs());
    mlir::Value derivedRhs = derivatives.lookup(rhs());

    auto derivedOp = builder.create<SubOp>(
        loc, convertToRealType(result().getType()), derivedLhs, derivedRhs);

    return derivedOp->getResults();
  }

  void SubOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(lhs());
    toBeDerived.push_back(rhs());
  }

  void SubOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // SubEWOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult SubEWOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto lhs = operands[0];
    auto rhs = operands[1];

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, getScalarIntegerLikeValue(lhs) - getScalarIntegerLikeValue(rhs));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, getScalarFloatLikeValue(lhs) - getScalarFloatLikeValue(rhs));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, getScalarIntegerLikeValue(lhs) - getScalarFloatLikeValue(rhs));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, getScalarFloatLikeValue(lhs) - getScalarIntegerLikeValue(rhs));
      }
    }

    return {};
  }

  void SubEWOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (lhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (rhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::LogicalResult SubEWOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (auto size = currentResult.size(); size != 1) {
      return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<AddEWOp>(getLoc(), lhs().getType(), nestedOperand, rhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(lhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<SubEWOp>(getLoc(), rhs().getType(), lhs(), nestedOperand);

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(rhs());
      erase();

      return mlir::success();
    }

    return emitError("Can't invert the operand #" + std::to_string(argumentIndex) + ". The operation has 2 operands.");
  }

  mlir::Value SubEWOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(definingOp)) {
          return casted.distributeNegateOp(builder, resultType);
        }
      }

      return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<SubEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value SubEWOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(definingOp)) {
          return casted.distributeMulOp(builder, resultType, value);
        }
      }

      return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<SubEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value SubEWOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      auto definingOp = child.getDefiningOp();

      if (definingOp != nullptr) {
        if (auto casted = mlir::dyn_cast<DivOpDistributionInterface>(definingOp)) {
          return casted.distributeDivOp(builder, resultType, value);
        }
      }

      return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<SubEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange SubEWOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(lhs());
    mlir::Value derivedRhs = derivatives.lookup(rhs());

    auto derivedOp = builder.create<SubEWOp>(
        loc, convertToRealType(result().getType()), derivedLhs, derivedRhs);

    return derivedOp->getResults();
  }

  void SubEWOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(lhs());
    toBeDerived.push_back(rhs());
  }

  void SubEWOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // SubscriptionOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange SubscriptionOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto derivedOp = builder.create<SubscriptionOp>(getLoc(), derivatives.lookup(source()), indices());
    return derivedOp->getResults();
  }

  void SubscriptionOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void SubscriptionOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // SymmetricOp
  //===----------------------------------------------------------------------===//

  void SymmetricOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), matrix(), mlir::SideEffects::DefaultResource::get());
    effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // TanOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult TanOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto operand = operands[0];

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        return getAttr(resultType, std::tan(getScalarIntegerLikeValue(operand)));
      }

      if (isScalarFloatLike(operand)) {
        return getAttr(resultType, std::tan(getScalarFloatLikeValue(operand)));
      }
    }

    return {};
  }

  void TanOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (operand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange TanOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int TanOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange TanOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<TanOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange TanOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[tan(x)] = x' / (cos(x))^2

    auto loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(operand());
    mlir::Type type = convertToRealType(result().getType());

    mlir::Value cos = builder.create<CosOp>(loc, type, operand());
    mlir::Value two = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 2));
    mlir::Value denominator = builder.create<PowEWOp>(loc, type, cos, two);
    auto derivedOp = builder.create<DivEWOp>(loc, type, derivedOperand, denominator);

    return derivedOp->getResults();
  }

  void TanOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void TanOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // TanhOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult TanhOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto operand = operands[0];

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        return getAttr(resultType, std::tanh(getScalarIntegerLikeValue(operand)));
      }

      if (isScalarFloatLike(operand)) {
        return getAttr(resultType, std::tanh(getScalarFloatLikeValue(operand)));
      }
    }

    return {};
  }

  void TanhOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (operand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange TanhOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int TanhOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange TanhOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<TanhOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange TanhOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[tanh(x)] = x' / (cosh(x))^2

    auto loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(operand());
    mlir::Type type = convertToRealType(result().getType());

    mlir::Value cosh = builder.create<CoshOp>(loc, type, operand());
    mlir::Value two = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 2));
    mlir::Value pow = builder.create<PowEWOp>(loc, type, cosh, two);
    auto derivedOp = builder.create<DivEWOp>(loc, type, derivedOperand, pow);

    return derivedOp->getResults();
  }

  void TanhOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void TanhOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // TransposeOp
  //===----------------------------------------------------------------------===//

  void TransposeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (matrix().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), matrix(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  //===----------------------------------------------------------------------===//
  // WhileOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange WhileOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    return llvm::None;
  }

  void WhileOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void WhileOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    regions.push_back(&bodyRegion());
  }

  //===----------------------------------------------------------------------===//
  // ZerosOp
  //===----------------------------------------------------------------------===//

  void ZerosOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // ArrayFillOp
  //===----------------------------------------------------------------------===//

  void ArrayFillOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (array().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Write::get(), array(), mlir::SideEffects::DefaultResource::get());
    }
  }

  //===----------------------------------------------------------------------===//
  // PrintOp
  //===----------------------------------------------------------------------===//

  void PrintOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (value().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), value(), mlir::SideEffects::DefaultResource::get());
    }
  }

  //===----------------------------------------------------------------------===//
  // EqOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult EqOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto lhs = operands[0];
    auto rhs = operands[1];

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarIntegerLikeValue(lhs) == getScalarIntegerLikeValue(rhs)));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarFloatLikeValue(lhs) == getScalarFloatLikeValue(rhs)));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarIntegerLikeValue(lhs) == getScalarFloatLikeValue(rhs)));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarFloatLikeValue(lhs) == getScalarIntegerLikeValue(rhs)));
      }
    }

    return {};
  }

  //===----------------------------------------------------------------------===//
  // NotEqOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult NotEqOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto lhs = operands[0];
    auto rhs = operands[1];

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarIntegerLikeValue(lhs) != getScalarIntegerLikeValue(rhs)));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarFloatLikeValue(lhs) != getScalarFloatLikeValue(rhs)));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarIntegerLikeValue(lhs) != getScalarFloatLikeValue(rhs)));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarFloatLikeValue(lhs) != getScalarIntegerLikeValue(rhs)));
      }
    }

    return {};
  }

  //===----------------------------------------------------------------------===//
  // GtOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult GtOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto lhs = operands[0];
    auto rhs = operands[1];

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarIntegerLikeValue(lhs) > getScalarIntegerLikeValue(rhs)));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarFloatLikeValue(lhs) > getScalarFloatLikeValue(rhs)));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarIntegerLikeValue(lhs) > getScalarFloatLikeValue(rhs)));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarFloatLikeValue(lhs) > getScalarIntegerLikeValue(rhs)));
      }
    }

    return {};
  }

  //===----------------------------------------------------------------------===//
  // GteOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult GteOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto lhs = operands[0];
    auto rhs = operands[1];

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarIntegerLikeValue(lhs) >= getScalarIntegerLikeValue(rhs)));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarFloatLikeValue(lhs) >= getScalarFloatLikeValue(rhs)));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarIntegerLikeValue(lhs) >= getScalarFloatLikeValue(rhs)));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarFloatLikeValue(lhs) >= getScalarIntegerLikeValue(rhs)));
      }
    }

    return {};
  }

  //===----------------------------------------------------------------------===//
  // LtOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult LtOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto lhs = operands[0];
    auto rhs = operands[1];

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarIntegerLikeValue(lhs) < getScalarIntegerLikeValue(rhs)));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarFloatLikeValue(lhs) < getScalarFloatLikeValue(rhs)));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarIntegerLikeValue(lhs) < getScalarFloatLikeValue(rhs)));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarFloatLikeValue(lhs) < getScalarIntegerLikeValue(rhs)));
      }
    }

    return {};
  }

  //===----------------------------------------------------------------------===//
  // LteOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult LteOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto lhs = operands[0];
    auto rhs = operands[1];

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarIntegerLikeValue(lhs) <= getScalarIntegerLikeValue(rhs)));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarFloatLikeValue(lhs) <= getScalarFloatLikeValue(rhs)));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarIntegerLikeValue(lhs) <= getScalarFloatLikeValue(rhs)));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        return getAttr(resultType, static_cast<long>(getScalarFloatLikeValue(lhs) <= getScalarIntegerLikeValue(rhs)));
      }
    }

    return {};
  }
}
