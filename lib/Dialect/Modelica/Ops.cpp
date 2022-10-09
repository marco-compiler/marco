#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/Modelica/Ops.h"
#include "marco/Codegen/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "llvm/ADT/MapVector.h"
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

  if (auto typedAttr = attribute.dyn_cast<mlir::TypedAttr>()) {
    return isScalar(typedAttr.getType());
  }

  return false;
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

  if (auto typedAttr = attribute.dyn_cast<mlir::TypedAttr>()) {
    return isScalarIntegerLike(typedAttr.getType());
  }

  return false;
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

  if (auto typedAttr = attribute.dyn_cast<mlir::TypedAttr>()) {
    return isScalarFloatLike(typedAttr.getType());
  }

  return false;
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

#define GET_OP_CLASSES
#include "marco/Dialect/Modelica/Modelica.cpp.inc"

namespace mlir::modelica
{
  //===----------------------------------------------------------------------===//
  // ModelOp
  //===----------------------------------------------------------------------===//

  mlir::ParseResult ModelOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::StringAttr nameAttr;

    if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes)) {
      return mlir::failure();
    }

    if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
      return mlir::failure();
    }

    mlir::Region* varsRegion = result.addRegion();
    mlir::Region* bodyRegion = result.addRegion();

    if (parser.parseRegion(*varsRegion)) {
      return mlir::failure();
    }

    if (mlir::succeeded(parser.parseOptionalKeyword("body"))) {
      if (parser.parseRegion(*bodyRegion)) {
        return mlir::failure();
      }
    }

    return mlir::success();
  }

  void ModelOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " ";
    printer.printSymbolName(getSymName());

    llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
    elidedAttrs.push_back(mlir::SymbolTable::getSymbolAttrName());

    printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(), elidedAttrs);

    printer << " ";
    printer.printRegion(getVarsRegion());

    if (auto& region = getBodyRegion(); !region.empty()) {
      printer << " body ";
      printer.printRegion(region);
    }
  }

  mlir::RegionKind ModelOp::getRegionKind(unsigned index)
  {
    if (index == 0) {
      return mlir::RegionKind::SSACFG;
    }

    return mlir::RegionKind::Graph;
  }

  SmallVector<StringRef> ModelOp::variableNames()
  {
    SmallVector<StringRef> result;

    for (MemberCreateOp member : getVarsRegion().getOps<MemberCreateOp>()) {
      result.push_back(member.getSymName());
    }

    return result;
  }

  mlir::Block* ModelOp::bodyBlock()
  {
    assert(getBodyRegion().hasOneBlock());
    return &getBodyRegion().front();
  }

  //===----------------------------------------------------------------------===//
  // FunctionOp
  //===----------------------------------------------------------------------===//

  mlir::ParseResult FunctionOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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
    result.attributes.append(getFunctionTypeAttrName(result.name), mlir::TypeAttr::get(functionType));

    if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
      return mlir::failure();
    }

    mlir::Region* bodyRegion = result.addRegion();

    if (parser.parseRegion(*bodyRegion, llvm::None)) {
      return mlir::failure();
    }

    return mlir::success();
  }

  void FunctionOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " ";
    printer.printSymbolName(getSymName());
    printer << " : " << getFunctionType();

    llvm::SmallVector<llvm::StringRef, 2> elidedAttrs;
    elidedAttrs.push_back(mlir::SymbolTable::getSymbolAttrName());
    elidedAttrs.push_back(mlir::function_interface_impl::getTypeAttrName());

    printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(), elidedAttrs);

    printer << " ";
    printer.printRegion(getBody());
  }

  mlir::Block* FunctionOp::bodyBlock()
  {
    assert(getBody().hasOneBlock());
    return &getBody().front();
  }

  SmallVector<StringRef> FunctionOp::inputMemberNames()
  {
    SmallVector<StringRef> result;

    walk<WalkOrder::PreOrder>([&](MemberCreateOp op) {
      if (op.isInput()) {
        result.push_back(op.getSymName());
      }
    });

    return result;
  }

  SmallVector<StringRef> FunctionOp::outputMemberNames()
  {
    SmallVector<StringRef> result;

    walk<WalkOrder::PreOrder>([&](MemberCreateOp op) {
      if (op.isOutput()) {
        result.push_back(op.getSymName());
      }
    });

    return result;
  }

  SmallVector<StringRef> FunctionOp::protectedMemberNames()
  {
    SmallVector<StringRef> result;

    walk<WalkOrder::PreOrder>([&](MemberCreateOp op) {
      if (!op.isInput() && !op.isOutput()) {
        result.push_back(op.getSymName());
      }
    });

    return result;
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
  // RawFunctionOp
  //===----------------------------------------------------------------------===//

  RawFunctionOp RawFunctionOp::create(
      mlir::Location location,
      llvm::StringRef name,
      mlir::FunctionType type,
      llvm::ArrayRef<mlir::NamedAttribute> attrs)
  {
    mlir::OpBuilder builder(location->getContext());
    mlir::OperationState state(location, getOperationName());
    RawFunctionOp::build(builder, state, name, type, attrs);
    return mlir::cast<RawFunctionOp>(mlir::Operation::create(state));
  }

  RawFunctionOp RawFunctionOp::create(
      mlir::Location location,
      llvm::StringRef name,
      mlir::FunctionType type,
      mlir::Operation::dialect_attr_range attrs)
  {
    llvm::SmallVector<mlir::NamedAttribute, 8> attrRef(attrs);
    return create(location, name, type, llvm::makeArrayRef(attrRef));
  }

  RawFunctionOp RawFunctionOp::create(
      mlir::Location location,
      llvm::StringRef name,
      mlir::FunctionType type,
      llvm::ArrayRef<mlir::NamedAttribute> attrs,
      llvm::ArrayRef<mlir::DictionaryAttr> argAttrs)
  {
    RawFunctionOp func = create(location, name, type, attrs);
    func.setAllArgAttrs(argAttrs);
    return func;
  }

  void RawFunctionOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      llvm::StringRef name,
      mlir::FunctionType type,
      llvm::ArrayRef<mlir::NamedAttribute> attrs,
      llvm::ArrayRef<mlir::DictionaryAttr> argAttrs)
  {
    state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
    state.addAttribute(mlir::FunctionOpInterface::getTypeAttrName(), mlir::TypeAttr::get(type));
    state.attributes.append(attrs.begin(), attrs.end());
    state.addRegion();

    if (argAttrs.empty()) {
      return;
    }

    assert(type.getNumInputs() == argAttrs.size());
    mlir::function_interface_impl::addArgAndResultAttrs(builder, state, argAttrs, llvm::None);
  }

  mlir::ParseResult RawFunctionOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto buildFuncType = [](mlir::Builder& builder, llvm::ArrayRef<mlir::Type> argTypes, llvm::ArrayRef<mlir::Type> results, mlir::function_interface_impl::VariadicFlag, std::string&) {
      return builder.getFunctionType(argTypes, results);
    };

    return mlir::function_interface_impl::parseFunctionOp(parser, result, false, buildFuncType);
  }

  void RawFunctionOp::print(OpAsmPrinter &p)
  {
    mlir::function_interface_impl::printFunctionOp(p, *this, false);
  }

  bool RawFunctionOp::shouldBeInlined()
  {
    if (!getOperation()->hasAttrOfType<mlir::BoolAttr>("inline")) {
      return false;
    }

    auto inlineAttribute = getOperation()->getAttrOfType<mlir::BoolAttr>("inline");
    return inlineAttribute.getValue();
  }

  /// Clone the internal blocks from this function into dest and all attributes
  /// from this function to dest.
  void RawFunctionOp::cloneInto(RawFunctionOp dest, mlir::BlockAndValueMapping& mapper)
  {
    // Add the attributes of this function to dest.
    llvm::MapVector<mlir::StringAttr, mlir::Attribute> newAttrMap;

    for (const auto &attr : dest->getAttrs()) {
      newAttrMap.insert({attr.getName(), attr.getValue()});
    }

    for (const auto &attr : (*this)->getAttrs()) {
      newAttrMap.insert({attr.getName(), attr.getValue()});
    }

    auto newAttrs = llvm::to_vector(llvm::map_range(
        newAttrMap, [](std::pair<StringAttr, Attribute> attrPair) {
          return NamedAttribute(attrPair.first, attrPair.second);
        }));

    dest->setAttrs(mlir::DictionaryAttr::get(getContext(), newAttrs));

    // Clone the body.
    getBody().cloneInto(&dest.getBody(), mapper);
  }

  /// Create a deep copy of this function and all of its blocks, remapping
  /// any operands that use values outside of the function using the map that is
  /// provided (leaving them alone if no entry is present). Replaces references
  /// to cloned sub-values with the corresponding value that is copied, and adds
  /// those mappings to the mapper.
  RawFunctionOp RawFunctionOp::clone(mlir::BlockAndValueMapping& mapper)
  {
    // Create the new function.
    RawFunctionOp newFunc = cast<RawFunctionOp>(getOperation()->cloneWithoutRegions());

    // If the function has a body, then the user might be deleting arguments to
    // the function by specifying them in the mapper. If so, we don't add the
    // argument to the input type vector.
    if (!isExternal()) {
      mlir::FunctionType oldType = getFunctionType();

      unsigned oldNumArgs = oldType.getNumInputs();
      llvm::SmallVector<Type, 4> newInputs;
      newInputs.reserve(oldNumArgs);

      for (unsigned i = 0; i != oldNumArgs; ++i) {
        if (!mapper.contains(getArgument(i))) {
          newInputs.push_back(oldType.getInput(i));
        }
      }

      // If any of the arguments were dropped, update the type and drop any
      // necessary argument attributes.
      if (newInputs.size() != oldNumArgs) {
        newFunc.setType(mlir::FunctionType::get(oldType.getContext(), newInputs, oldType.getResults()));

        if (ArrayAttr argAttrs = getAllArgAttrs()) {
          SmallVector<Attribute> newArgAttrs;
          newArgAttrs.reserve(newInputs.size());

          for (unsigned i = 0; i != oldNumArgs; ++i) {
            if (!mapper.contains(getArgument(i))) {
              newArgAttrs.push_back(argAttrs[i]);
            }
          }

          newFunc.setAllArgAttrs(newArgAttrs);
        }
      }
    }

    // Clone the current function into the new one and return it.
    cloneInto(newFunc, mapper);
    return newFunc;
  }

  RawFunctionOp RawFunctionOp::clone()
  {
    mlir::BlockAndValueMapping mapper;
    return clone(mapper);
  }

  //===----------------------------------------------------------------------===//
  // RawReturnOp
  //===----------------------------------------------------------------------===//

  mlir::LogicalResult RawReturnOp::verify() {
    auto function = cast<RawFunctionOp>((*this)->getParentOp());

    // The operand number and types must match the function signature
    const auto& results = function.getFunctionType().getResults();

    if (getNumOperands() != results.size()) {
      return emitOpError("has ")
          << getNumOperands() << " operands, but enclosing function (@"
          << function.getName() << ") returns " << results.size();
    }

    for (unsigned i = 0, e = results.size(); i != e; ++i) {
      if (getOperand(i).getType() != results[i]) {
        return emitError() << "type of return operand " << i << " ("
                           << getOperand(i).getType()
                           << ") doesn't match function result type ("
                           << results[i] << ")"
                           << " in function @" << function.getName();
      }
    }

    return mlir::success();
  }

  //===----------------------------------------------------------------------===//
  // RuntimeFunctionOp
  //===----------------------------------------------------------------------===//

  void RuntimeFunctionOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      llvm::StringRef name,
      mlir::FunctionType type,
      llvm::ArrayRef<mlir::NamedAttribute> attrs,
      llvm::ArrayRef<mlir::DictionaryAttr> argAttrs)
  {
    state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
    state.addAttribute(mlir::FunctionOpInterface::getTypeAttrName(), mlir::TypeAttr::get(type));
    state.addAttribute(mlir::SymbolTable::getVisibilityAttrName(), builder.getStringAttr("private"));
    state.attributes.append(attrs.begin(), attrs.end());

    if (argAttrs.empty()) {
      return;
    }

    assert(type.getNumInputs() == argAttrs.size());
    mlir::function_interface_impl::addArgAndResultAttrs(builder, state, argAttrs, llvm::None);
  }

  mlir::ParseResult RuntimeFunctionOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto& builder = parser.getBuilder();
    mlir::StringAttr nameAttr;

    if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes)) {
      return mlir::failure();
    }

    mlir::Type functionType;

    if (parser.parseColon() ||
        parser.parseType(functionType)) {
      return mlir::failure();
    }

    result.attributes.append(mlir::function_interface_impl::getTypeAttrName(), mlir::TypeAttr::get(functionType));
    result.attributes.append(mlir::SymbolTable::getVisibilityAttrName(), builder.getStringAttr("private"));

    if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
      return mlir::failure();
    }

    return mlir::success();
  }

  void RuntimeFunctionOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " ";
    printer.printSymbolName(getSymName());

    llvm::SmallVector<llvm::StringRef, 3> elidedAttrs;
    elidedAttrs.push_back(mlir::SymbolTable::getSymbolAttrName());
    elidedAttrs.push_back(mlir::SymbolTable::getVisibilityAttrName());
    elidedAttrs.push_back(mlir::function_interface_impl::getTypeAttrName());

    printer.printOptionalAttrDict(getOperation()->getAttrs(), elidedAttrs);
    printer << " : " << getFunctionType();
  }

  //===----------------------------------------------------------------------===//
  // EquationOp
  //===----------------------------------------------------------------------===//

  mlir::Block* EquationOp::bodyBlock()
  {
    assert(getBodyRegion().getBlocks().size() == 1);
    return &getBodyRegion().front();
  }

  mlir::ParseResult EquationOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

  void EquationOp::print(mlir::OpAsmPrinter& printer)
  {
    printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
    printer << " ";
    printer.printRegion(getBodyRegion());
  }

  //===----------------------------------------------------------------------===//
  // InitialEquationOp
  //===----------------------------------------------------------------------===//

  mlir::Block* InitialEquationOp::bodyBlock()
  {
    assert(getBodyRegion().getBlocks().size() == 1);
    return &getBodyRegion().front();
  }

  mlir::ParseResult InitialEquationOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

  void InitialEquationOp::print(mlir::OpAsmPrinter& printer)
  {
    printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
    printer << " ";
    printer.printRegion(getBodyRegion());
  }

  //===----------------------------------------------------------------------===//
  // ForEquationOp
  //===----------------------------------------------------------------------===//

  void ForEquationOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, long from, long to, long step)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    state.addAttribute(getFromAttrName(state.name), builder.getIndexAttr(from));
    state.addAttribute(getToAttrName(state.name), builder.getIndexAttr(to));
    state.addAttribute(getStepAttrName(state.name), builder.getIndexAttr(step));

    mlir::Region* bodyRegion = state.addRegion();
    builder.createBlock(bodyRegion, {}, builder.getIndexType(), builder.getUnknownLoc());
  }

  mlir::Block* ForEquationOp::bodyBlock()
  {
    assert(getBodyRegion().getBlocks().size() == 1);
    return &getBodyRegion().front();
  }

  mlir::Value ForEquationOp::induction()
  {
    assert(getBodyRegion().getNumArguments() != 0);
    return getBodyRegion().getArgument(0);
  }

  mlir::ParseResult ForEquationOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto& builder = parser.getBuilder();

    mlir::OpAsmParser::Argument induction;

    long from;
    long to;
    long step = 1;

    if (parser.parseArgument(induction) ||
        parser.parseEqual() ||
        parser.parseInteger(from) ||
        parser.parseKeyword("to") ||
        parser.parseInteger(to)) {
      return mlir::failure();
    }

    if (mlir::succeeded(parser.parseOptionalKeyword("step"))) {
      if (parser.parseInteger(step)) {
        return mlir::failure();
      }
    }

    induction.type = builder.getIndexType();

    result.attributes.append("from", builder.getIndexAttr(from));
    result.attributes.append("to", builder.getIndexAttr(to));
    result.attributes.append("step", builder.getIndexAttr(step));

    if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
      return mlir::failure();
    }

    mlir::Region* bodyRegion = result.addRegion();

    if (parser.parseRegion(*bodyRegion, induction)) {
      return mlir::failure();
    }

    return mlir::success();
  }

  void ForEquationOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " " << induction() << " = " << getFrom() << " to " << getTo();

    if (auto step = getStep(); step != 1) {
      printer << " step " << step;
    }

    printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(), {"from", "to", "step"});
    printer << " ";
    printer.printRegion(getBodyRegion(), false);
  }

  //===----------------------------------------------------------------------===//
  // AlgorithmOp
  //===----------------------------------------------------------------------===//

  mlir::Block* AlgorithmOp::bodyBlock()
  {
    assert(getBodyRegion().getBlocks().size() == 1);
    return &getBodyRegion().front();
  }

  mlir::ParseResult AlgorithmOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

  void AlgorithmOp::print(mlir::OpAsmPrinter& printer)
  {
    printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
    printer << " ";
    printer.printRegion(getBodyRegion());
  }

  //===----------------------------------------------------------------------===//
  // InitialAlgorithmOp
  //===----------------------------------------------------------------------===//

  mlir::Block* InitialAlgorithmOp::bodyBlock()
  {
    assert(getBodyRegion().getBlocks().size() == 1);
    return &getBodyRegion().front();
  }

  mlir::ParseResult InitialAlgorithmOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

  void InitialAlgorithmOp::print(mlir::OpAsmPrinter& printer)
  {
    printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
    printer << " ";
    printer.printRegion(getBodyRegion());
  }

  //===----------------------------------------------------------------------===//
  // ConstantOp
  //===----------------------------------------------------------------------===//

  mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::Attribute value;

    if (parser.parseAttribute(value)) {
      return mlir::failure();
    }

    result.attributes.append("value", value);
    result.addTypes(value.cast<mlir::TypedAttr>().getType());

    return mlir::success();
  }

  void ConstantOp::print(mlir::OpAsmPrinter& printer)
  {
    printer.printOptionalAttrDict(getOperation()->getAttrs(), {"value"});
    printer << " " << getValue();

    // If the value is a symbol reference, print a trailing type.
    if (getValue().isa<mlir::SymbolRefAttr>()) {
      printer << " : " << getType();
    }
  }

  mlir::OpFoldResult ConstantOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    return getValue();
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
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getOperand(), mlir::SideEffects::DefaultResource::get());
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

    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value sqrt = builder.create<SqrtOp>(loc, type, getOperand());
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
  // MemberCreateOp
  //===----------------------------------------------------------------------===//

  mlir::ParseResult MemberCreateOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto& builder = parser.getBuilder();
    mlir::StringAttr nameAttr;

    if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes)) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 1> dynamicSizes;
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

  void MemberCreateOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " ";
    printer.printSymbolName(getSymName());

    llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
    elidedAttrs.push_back(mlir::SymbolTable::getSymbolAttrName());

    printer.printOptionalAttrDict(getOperation()->getAttrs(), elidedAttrs);

    printer << getDynamicSizes();
    printer << " : " << getResult().getType();
  }

  //===----------------------------------------------------------------------===//
  // MemberLoadOp
  //===----------------------------------------------------------------------===//

  mlir::ParseResult MemberLoadOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::OpAsmParser::UnresolvedOperand member;
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

  void MemberLoadOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " ";
    printer << getMember();
    printer.printOptionalAttrDict(getOperation()->getAttrs());
    printer << " : " << getMember().getType();

    if (auto resultType = getResult().getType(); resultType != getMemberType().unwrap()) {
      printer << " -> " << resultType;
    }
  }

  mlir::ValueRange MemberLoadOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto derivedOp = builder.create<MemberLoadOp>(getLoc(), getResult().getType(), derivatives.lookup(getMember()));
    return derivedOp->getResults();
  }

  void MemberLoadOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getMember());
  }

  void MemberLoadOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // MemberStoreOp
  //===----------------------------------------------------------------------===//

  mlir::ParseResult MemberStoreOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::OpAsmParser::UnresolvedOperand member;
    mlir::OpAsmParser::UnresolvedOperand value;
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

  void MemberStoreOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " ";
    printer << getMember() << ", " << getValue();
    printer.printOptionalAttrDict(getOperation()->getAttrs());
    printer << " : " << getMember().getType();

    if (auto valueType = getValue().getType(); valueType != getMemberType().unwrap()) {
      printer << ", " << valueType;
    }
  }

  mlir::ValueRange MemberStoreOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // Store operations should be derived only if they store a value into
    // a member whose derivative is created by the current function. Otherwise,
    // we would create a double store into that derived member.

    assert(derivatives.contains(getMember()) && "Derived member not found");
    mlir::Value derivedMember = derivatives.lookup(getMember());
    builder.create<MemberStoreOp>(getLoc(), derivedMember, derivatives.lookup(getValue()));

    return llvm::None;
  }

  void MemberStoreOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getValue());
    toBeDerived.push_back(getMember());
  }

  void MemberStoreOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // StartOp
  //===----------------------------------------------------------------------===//

  mlir::ParseResult StartOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::OpAsmParser::UnresolvedOperand variable;
    mlir::Type variableType;

    mlir::Region* bodyRegion = result.addRegion();

    if (parser.parseLParen() ||
        parser.parseOperand(variable) ||
        parser.parseColonType(variableType) ||
        parser.parseRParen() ||
        parser.resolveOperand(variable, variableType, result.operands) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseRegion(*bodyRegion)) {
      return mlir::failure();
    }

    return mlir::success();
  }

  void StartOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " ";
    printer << "(" << getVariable() << " : " << getVariable().getType() << ")";
    printer.printOptionalAttrDict(getOperation()->getAttrs());
    printer << " ";
    printer.printRegion(getBodyRegion());
  }

  //===----------------------------------------------------------------------===//
  // CastOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange CastOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto derivedOp = builder.create<CastOp>(getLoc(), getResult().getType(), derivatives.lookup(getValue()));
    return derivedOp->getResults();
  }

  void CastOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getValue());
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

  unsigned int CallOp::getArgExpectedRank(unsigned int argIndex)
  {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto function = module.lookupSymbol<FunctionOp>(getCallee());

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

    for (mlir::Value arg : getArgs())
    {
      assert(arg.getType().isa<ArrayType>());
      mlir::Value newArg = builder.create<SubscriptionOp>(getLoc(), arg, indexes);

      if (auto arrayType = newArg.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
        newArg = builder.create<LoadOp>(getLoc(), newArg);

      newArgs.push_back(newArg);
    }

    auto op = builder.create<CallOp>(getLoc(), getCallee(), newResultsTypes, newArgs);
    return op->getResults();
  }

  mlir::LogicalResult CallOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (getNumResults() != 1) {
      return emitError("The callee must have one and only one result");
    }

    if (argumentIndex >= getArgs().size()) {
      return emitError("Index out of bounds: " + std::to_string(argumentIndex));
    }

    if (auto size = currentResult.size(); size != 1) {
      return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");
    }

    mlir::Value toNest = currentResult[0];

    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto callee = module.lookupSymbol<FunctionOp>(this->getCallee());

    if (!callee->hasAttr("inverse")) {
      return emitError("Function " + callee->getName().getStringRef() + " is not invertible");
    }

    auto inverseAnnotation = callee->getAttrOfType<InverseFunctionsAttr>("inverse");

    if (!inverseAnnotation.isInvertible(argumentIndex)) {
      return emitError("Function " + callee->getName().getStringRef() + " is not invertible for argument " + std::to_string(argumentIndex));
    }

    size_t argsSize = getArgs().size();
    llvm::SmallVector<mlir::Value, 3> args;

    for (auto arg : inverseAnnotation.getArgumentsIndexes(argumentIndex)) {
      if (arg < argsSize) {
        args.push_back(this->getArgs()[arg]);
      } else {
        assert(arg == argsSize);
        args.push_back(toNest);
      }
    }

    auto invertedCall = builder.create<CallOp>(getLoc(), inverseAnnotation.getFunction(argumentIndex), this->getArgs()[argumentIndex].getType(), args);

    getResult(0).replaceAllUsesWith(this->getArgs()[argumentIndex]);
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

    mlir::Value derivedY = derivatives.lookup(getY());
    mlir::Value derivedX = derivatives.lookup(getX());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value firstMul = builder.create<MulEWOp>(loc, type, derivedY, getX());
    mlir::Value secondMul = builder.create<MulEWOp>(loc, type, getY(), derivedX);
    mlir::Value numerator = builder.create<SubEWOp>(loc, type, firstMul, secondMul);

    mlir::Value firstSquared = builder.create<MulEWOp>(loc, type, getY(), getY());
    mlir::Value secondSquared = builder.create<MulEWOp>(loc, type, getX(), getX());
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
    if (getValue().getType().isa<ArrayType>())
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getValue(), mlir::SideEffects::DefaultResource::get());

    if (getDestination().getType().isa<ArrayType>())
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getDestination(), mlir::SideEffects::DefaultResource::get());
  }

  mlir::ValueRange AssignmentOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto loc = getLoc();

    mlir::Value derivedSource = derivatives.lookup(getValue());
    mlir::Value derivedDestination = derivatives.lookup(getDestination());

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

  mlir::LogicalResult AllocaOp::verify()
  {
    auto dynamicDimensionsAmount = getArrayType().getNumDynamicDims();
    auto valuesAmount = getDynamicSizes().size();

    if (static_cast<size_t>(dynamicDimensionsAmount) != valuesAmount) {
      return emitOpError(
          "incorrect number of values for dynamic dimensions (expected " +
          std::to_string(dynamicDimensionsAmount) + ", got " +
          std::to_string(valuesAmount) + ")");
    }

    return mlir::success();
  }

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

  mlir::LogicalResult AllocOp::verify()
  {
    auto dynamicDimensionsAmount = getArrayType().getNumDynamicDims();
    auto valuesAmount = getDynamicSizes().size();

    if (static_cast<size_t>(dynamicDimensionsAmount) != valuesAmount) {
      return emitOpError(
          "incorrect number of values for dynamic dimensions (expected " +
          std::to_string(dynamicDimensionsAmount) + ", got " +
          std::to_string(valuesAmount) + ")");
    }

    return mlir::success();
  }

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

  mlir::ParseResult DerFunctionOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::StringAttr nameAttr;

    if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes) ||
        parser.parseOptionalAttrDict(result.attributes)) {
      return mlir::failure();
    }

    return mlir::success();
  }

  void DerFunctionOp::print(mlir::OpAsmPrinter& printer)
  {
    printer.printSymbolName(getSymName());

    llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
    elidedAttrs.push_back(mlir::SymbolTable::getSymbolAttrName());

    printer.printOptionalAttrDict(getOperation()->getAttrs(), elidedAttrs);
  }

  mlir::ArrayRef<mlir::Type> DerFunctionOp::getCallableResults()
  {
    auto module = getOperation()->getParentOfType<::mlir::ModuleOp>();
    return mlir::cast<mlir::CallableOpInterface>(module.lookupSymbol(getDerivedFunction())).getCallableResults();
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
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getOperand(), mlir::SideEffects::DefaultResource::get());
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
    mlir::Type newResultType = getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), getOperand(), indexes);

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
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getOperand(), mlir::SideEffects::DefaultResource::get());
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
    mlir::Type newResultType = getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), getOperand(), indexes);

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
    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value one = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 1));
    mlir::Value two = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 2));
    mlir::Value argSquared = builder.create<PowEWOp>(loc, type, getOperand(), two);
    mlir::Value sub = builder.create<SubEWOp>(loc, type, one, argSquared);
    mlir::Value denominator = builder.create<SqrtOp>(loc, type, sub);
    mlir::Value div = builder.create<DivEWOp>(loc, type, derivedOperand, denominator);
    auto derivedOp = builder.create<NegateOp>(loc, type, div);

    return derivedOp->getResults();
  }

  void AcosOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
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
    if (getLhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getLhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (getRhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getRhs(), mlir::SideEffects::DefaultResource::get());
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
      auto right = builder.create<SubOp>(getLoc(), getLhs().getType(), nestedOperand, getRhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(getLhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<SubOp>(getLoc(), getRhs().getType(), nestedOperand, getLhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(getRhs());
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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = distributeFn(this->getRhs());

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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = distributeFn(this->getRhs());

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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = distributeFn(this->getRhs());

    return builder.create<AddOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange AddOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(getLhs());
    mlir::Value derivedRhs = derivatives.lookup(getRhs());

    auto derivedOp = builder.create<AddOp>(
        loc, convertToRealType(getResult().getType()), derivedLhs, derivedRhs);

    return derivedOp->getResults();
  }

  void AddOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getLhs());
    toBeDerived.push_back(getRhs());
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
    if (getLhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getLhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (getRhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getRhs(), mlir::SideEffects::DefaultResource::get());
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
      auto right = builder.create<SubEWOp>(getLoc(), getLhs().getType(), nestedOperand, getRhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(getLhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<SubEWOp>(getLoc(), getRhs().getType(), nestedOperand, getLhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(getRhs());
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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = distributeFn(this->getRhs());

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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = distributeFn(this->getRhs());

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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = distributeFn(this->getRhs());

    return builder.create<AddEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange AddEWOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(getLhs());
    mlir::Value derivedRhs = derivatives.lookup(getRhs());

    auto derivedOp = builder.create<AddEWOp>(
        loc, convertToRealType(getResult().getType()), derivedLhs, derivedRhs);

    return derivedOp->getResults();
  }

  void AddEWOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getLhs());
    toBeDerived.push_back(getRhs());
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
    if (getLhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getLhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (getRhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getRhs(), mlir::SideEffects::DefaultResource::get());
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
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getOperand(), mlir::SideEffects::DefaultResource::get());
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
    mlir::Type newResultType = getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), getOperand(), indexes);

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
    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value one = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 1));
    mlir::Value two = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 2));
    mlir::Value argSquared = builder.create<PowEWOp>(loc, type, getOperand(), two);
    mlir::Value sub = builder.create<SubEWOp>(loc, type, one, argSquared);
    mlir::Value denominator = builder.create<SqrtOp>(loc, type, sub);
    auto derivedOp = builder.create<DivEWOp>(loc, type, derivedOperand, denominator);

    return derivedOp->getResults();
  }

  void AsinOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
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
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getOperand(), mlir::SideEffects::DefaultResource::get());
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
    mlir::Type newResultType = getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), getOperand(), indexes);

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
    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value one = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 1));
    mlir::Value two = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 2));
    mlir::Value argSquared = builder.create<PowEWOp>(loc, type, getOperand(), two);
    mlir::Value denominator = builder.create<AddEWOp>(loc, type, one, argSquared);
    auto derivedOp = builder.create<DivEWOp>(loc, type, derivedOperand, denominator);

    return derivedOp->getResults();
  }

  void AtanOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
  }

  void AtanOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // Atan2Op
  //===----------------------------------------------------------------------===//

  void Atan2Op::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (getY().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getY(), mlir::SideEffects::DefaultResource::get());
    }

    if (getX().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getX(), mlir::SideEffects::DefaultResource::get());
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
    mlir::Type newResultType = getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newY = builder.create<SubscriptionOp>(getLoc(), getY(), indexes);

    if (auto arrayType = newY.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newY = builder.create<LoadOp>(getLoc(), newY);
    }

    mlir::Value newX = builder.create<SubscriptionOp>(getLoc(), getX(), indexes);

    if (auto arrayType = newX.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newX = builder.create<LoadOp>(getLoc(), newX);
    }

    auto op = builder.create<Atan2Op>(getLoc(), newResultType, newY, newX);
    return op->getResults();
  }

  //===----------------------------------------------------------------------===//
  // CeilOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult CeilOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto operand = operands[0];

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        return getAttr(resultType, getScalarIntegerLikeValue(operand));
      }

      if (isScalarFloatLike(operand)) {
        return getAttr(resultType, std::ceil(getScalarFloatLikeValue(operand)));
      }
    }

    return {};
  }

  void CeilOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getOperand(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange CeilOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int CeilOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange CeilOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<CeilOp>(getLoc(), newResultType, newOperand);
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
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getOperand(), mlir::SideEffects::DefaultResource::get());
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
    mlir::Type newResultType = getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), getOperand(), indexes);

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
    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value sin = builder.create<SinOp>(loc, type, getOperand());
    mlir::Value negatedSin = builder.create<NegateOp>(loc, type, sin);
    auto derivedOp = builder.create<MulEWOp>(loc, type, negatedSin, derivedOperand);

    return derivedOp->getResults();
  }

  void CosOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
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
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getOperand(), mlir::SideEffects::DefaultResource::get());
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
    mlir::Type newResultType = getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), getOperand(), indexes);

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
    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value sinh = builder.create<SinhOp>(loc, type, getOperand());
    auto derivedOp = builder.create<MulEWOp>(loc, type, sinh, derivedOperand);

    return derivedOp->getResults();
  }

  void CoshOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
  }

  void CoshOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // DiagonalOp
  //===----------------------------------------------------------------------===//

  void DiagonalOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), getValues(), mlir::SideEffects::DefaultResource::get());
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
    if (getLhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getLhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (getRhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getRhs(), mlir::SideEffects::DefaultResource::get());
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
      auto right = builder.create<MulOp>(getLoc(), getLhs().getType(), nestedOperand, getRhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      getResult().replaceAllUsesWith(getLhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<DivOp>(getLoc(), getRhs().getType(), getLhs(), nestedOperand);

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      getResult().replaceAllUsesWith(getRhs());
      erase();

      return mlir::success();
    }

    return emitError("Index out of bounds: " + std::to_string(argumentIndex));
  }

  mlir::Value DivOp::distribute(mlir::OpBuilder& builder)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(*this);

    auto lhsDefiningOp = getLhs().getDefiningOp();

    if (!lhsDefiningOp) {
      return getResult();
    }

    if (!mlir::isa<DivOpDistributionInterface>(lhsDefiningOp)) {
      // The operation can't be propagated because the dividend does not
      // know how to distribute the division to their children.
      return getResult();
    }

    DivOpDistributionInterface childOp = mlir::cast<DivOpDistributionInterface>(lhsDefiningOp);
    mlir::Value toDistribute = getRhs();

    return childOp.distributeDivOp(builder, getResult().getType(), toDistribute);
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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = this->getRhs();

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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = this->getRhs();

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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = this->getRhs();

    return builder.create<DivOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange DivOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(getLhs());
    mlir::Value derivedRhs = derivatives.lookup(getRhs());

    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value firstMul = builder.create<MulOp>(loc, type, derivedLhs, getRhs());
    mlir::Value secondMul = builder.create<MulOp>(loc, type, getLhs(), derivedRhs);
    mlir::Value numerator = builder.create<SubOp>(loc, type, firstMul, secondMul);
    mlir::Value two = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 2));
    mlir::Value denominator = builder.create<PowOp>(loc, convertToRealType(getRhs().getType()), getRhs(), two);
    auto derivedOp = builder.create<DivOp>(loc, type, numerator, denominator);

    return derivedOp->getResults();
  }

  void DivOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getLhs());
    toBeDerived.push_back(getRhs());
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
    if (getLhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getLhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (getRhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getRhs(), mlir::SideEffects::DefaultResource::get());
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
      auto right = builder.create<MulEWOp>(getLoc(), getLhs().getType(), nestedOperand, getRhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      getResult().replaceAllUsesWith(getLhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<DivEWOp>(getLoc(), getRhs().getType(), getLhs(), nestedOperand);

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      getResult().replaceAllUsesWith(getRhs());
      erase();

      return mlir::success();
    }

    return emitError("Index out of bounds: " + std::to_string(argumentIndex));
  }

  mlir::Value DivEWOp::distribute(mlir::OpBuilder& builder)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(*this);

    auto lhsDefiningOp = getLhs().getDefiningOp();

    if (!lhsDefiningOp) {
      return getResult();
    }

    if (!mlir::isa<DivOpDistributionInterface>(lhsDefiningOp)) {
      // The operation can't be propagated because the dividend does not
      // know how to distribute the division to their children.
      return getResult();
    }

    DivOpDistributionInterface childOp = mlir::cast<DivOpDistributionInterface>(lhsDefiningOp);
    mlir::Value toDistribute = getRhs();

    return childOp.distributeDivOp(builder, getResult().getType(), toDistribute);
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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = this->getRhs();

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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = this->getRhs();

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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = this->getRhs();

    return builder.create<DivEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange DivEWOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(getLhs());
    mlir::Value derivedRhs = derivatives.lookup(getRhs());

    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value firstMul = builder.create<MulEWOp>(loc, type, derivedLhs, getRhs());
    mlir::Value secondMul = builder.create<MulEWOp>(loc, type, getLhs(), derivedRhs);
    mlir::Value numerator = builder.create<SubEWOp>(loc, type, firstMul, secondMul);
    mlir::Value two = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 2));
    mlir::Value denominator = builder.create<PowEWOp>(loc, convertToRealType(getRhs().getType()), getRhs(), two);
    auto derivedOp = builder.create<DivEWOp>(loc, type, numerator, denominator);

    return derivedOp->getResults();
  }

  void DivEWOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getLhs());
    toBeDerived.push_back(getRhs());
  }

  void DivEWOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // DivTruncOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult DivTruncOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto x = operands[0];
    auto y = operands[1];

    if (!x || !y) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(x) && isScalar(y)) {
      if (isScalarIntegerLike(x) && isScalarIntegerLike(y)) {
        auto xValue = getScalarIntegerLikeValue(x);
        auto yValue = getScalarIntegerLikeValue(y);
        return getAttr(resultType, xValue / yValue);
      }

      if (isScalarFloatLike(x) && isScalarFloatLike(y)) {
        auto xValue = getScalarFloatLikeValue(x);
        auto yValue = getScalarFloatLikeValue(y);
        return getAttr(resultType, std::trunc(xValue / yValue));
      }

      if (isScalarIntegerLike(x) && isScalarFloatLike(y)) {
        auto xValue = getScalarIntegerLikeValue(x);
        auto yValue = getScalarFloatLikeValue(y);
        return getAttr(resultType, std::trunc(xValue / yValue));
      }

      if (isScalarFloatLike(x) && isScalarIntegerLike(y)) {
        auto xValue = getScalarFloatLikeValue(x);
        auto yValue = getScalarIntegerLikeValue(y);
        return getAttr(resultType, std::trunc(xValue / yValue));
      }
    }

    return {};
  }

  //===----------------------------------------------------------------------===//
  // SelectOp
  //===----------------------------------------------------------------------===//

  mlir::ParseResult SelectOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::OpAsmParser::UnresolvedOperand condition;
    mlir::Type conditionType;

    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 1> trueValues;
    llvm::SmallVector<mlir::Type, 1> trueValuesTypes;

    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 1> falseValues;
    llvm::SmallVector<mlir::Type, 1> falseValuesTypes;

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    if (parser.parseLParen() ||
        parser.parseOperand(condition) ||
        parser.parseColonType(conditionType) ||
        parser.parseRParen() ||
        parser.resolveOperand(condition, conditionType, result.operands)) {
      return mlir::failure();
    }

    if (parser.parseComma()) {
      return mlir::failure();
    }

    auto trueValuesLoc = parser.getCurrentLocation();

    if (parser.parseLParen() ||
        parser.parseOperandList(trueValues) ||
        parser.parseColonTypeList(trueValuesTypes) ||
        parser.parseRParen() ||
        parser.resolveOperands(trueValues, trueValuesTypes, trueValuesLoc, result.operands)) {
      return mlir::failure();
    }

    if (parser.parseComma()) {
      return mlir::failure();
    }

    auto falseValuesLoc = parser.getCurrentLocation();

    if (parser.parseLParen() ||
        parser.parseOperandList(falseValues) ||
        parser.parseColonTypeList(falseValuesTypes) ||
        parser.parseRParen() ||
        parser.resolveOperands(falseValues, falseValuesTypes, falseValuesLoc, result.operands)) {
      return mlir::failure();
    }

    if (parser.parseArrowTypeList(resultTypes)) {
      return mlir::failure();
    }

    result.addTypes(resultTypes);
    return mlir::success();
  }

  void SelectOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " ";
    printer << "(" << getCondition() << " : " << getCondition().getType() << ")";
    printer << ", ";
    printer << "(" <<  getTrueValues() << " : " << getTrueValues().getTypes() << ")";
    printer << ", ";
    printer << "(" <<  getFalseValues() << " : " << getFalseValues().getTypes() << ")";
    printer.printOptionalAttrDict(getOperation()->getAttrs());
    printer << " -> ";

    auto resultTypes = getResultTypes();

    if (resultTypes.size() == 1) {
      printer << resultTypes;
    } else {
      printer << "(" << resultTypes << ")";
    }
  }

  //===----------------------------------------------------------------------===//
  // EquationSideOp
  //===----------------------------------------------------------------------===//

  mlir::ParseResult EquationSideOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 1> values;
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

  void EquationSideOp::print(mlir::OpAsmPrinter& printer)
  {
    printer.printOptionalAttrDict(getOperation()->getAttrs());
    printer << " " << getValues() << " : " << getResult().getType();
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
    if (getExponent().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getExponent(), mlir::SideEffects::DefaultResource::get());
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
    mlir::Type newResultType = getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), getExponent(), indexes);

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
    mlir::Value derivedExponent = derivatives.lookup(getExponent());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value pow = builder.create<ExpOp>(loc, type, getExponent());
    auto derivedOp = builder.create<MulEWOp>(loc, type, pow, derivedExponent);

    return derivedOp->getResults();
  }

  void ExpOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getExponent());
  }

  void ExpOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // FloorOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult FloorOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto operand = operands[0];

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        return getAttr(resultType, getScalarIntegerLikeValue(operand));
      }

      if (isScalarFloatLike(operand)) {
        return getAttr(resultType, std::floor(getScalarFloatLikeValue(operand)));
      }
    }

    return {};
  }

  void FloorOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getOperand(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange FloorOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int FloorOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange FloorOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<FloorOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  //===----------------------------------------------------------------------===//
  // ForOp
  //===----------------------------------------------------------------------===//

  mlir::Block* ForOp::conditionBlock()
  {
    assert(!getConditionRegion().empty());
    return &getConditionRegion().front();
  }

  mlir::Block* ForOp::bodyBlock()
  {
    assert(!getBodyRegion().empty());
    return &getBodyRegion().front();
  }

  mlir::Block* ForOp::stepBlock()
  {
    assert(!getStepRegion().empty());
    return &getStepRegion().front();
  }

  mlir::ParseResult ForOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::Region* conditionRegion = result.addRegion();

    if (mlir::succeeded(parser.parseOptionalLParen())) {
      if (mlir::failed(parser.parseOptionalRParen())) {
        do {
          mlir::OpAsmParser::UnresolvedOperand arg;
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

  void ForOp::print(mlir::OpAsmPrinter& printer)
  {
    if (auto values = getArgs(); !values.empty()) {
      printer << "(";

      for (auto arg : llvm::enumerate(values)) {
        if (arg.index() != 0) {
          printer << ", ";
        }

        printer << arg.value() << " : " << arg.value().getType();
      }

      printer << ")";
    }

    printer << " condition";
    printer.printRegion(getConditionRegion(), true);
    printer << " body";
    printer.printRegion(getBodyRegion(), true);
    printer << " step";
    printer.printRegion(getStepRegion(), true);
    printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
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
    regions.push_back(&getBodyRegion());
  }

  //===----------------------------------------------------------------------===//
  // FreeOp
  //===----------------------------------------------------------------------===//

  void FreeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Free::get(), getArray(), mlir::SideEffects::DefaultResource::get());
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
    return &getThenRegion().front();
  }

  mlir::Block* IfOp::elseBlock()
  {
    return &getElseRegion().front();
  }

  mlir::ParseResult IfOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::OpAsmParser::UnresolvedOperand condition;
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

  void IfOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << "(" << getCondition() << " : " << getCondition().getType() << ")";

    printer.printRegion(getThenRegion());

    if (!getElseRegion().empty()) {
      printer << " else";
      printer.printRegion(getElseRegion());
    }

    printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
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
    regions.push_back(&getThenRegion());
    regions.push_back(&getElseRegion());
  }

  //===----------------------------------------------------------------------===//
  // IntegerOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult IntegerOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto operand = operands[0];

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        return getAttr(resultType, getScalarIntegerLikeValue(operand));
      }

      if (isScalarFloatLike(operand)) {
        return getAttr(resultType, std::floor(getScalarFloatLikeValue(operand)));
      }
    }

    return {};
  }

  void IntegerOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getOperand(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange IntegerOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int IntegerOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange IntegerOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<FloorOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
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

  mlir::LogicalResult LoadOp::verify()
  {
    auto indicesAmount = getIndices().size();
    auto rank = getArrayType().getRank();

    if (indicesAmount != static_cast<size_t>(rank)) {
      return emitOpError(
          "incorrect number of indices for load (expected " +
          std::to_string(rank) + ", got " + std::to_string(indicesAmount) + ")");
    }

    return mlir::success();
  }

  void LoadOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), getArray(), mlir::SideEffects::DefaultResource::get());
  }

  mlir::ValueRange LoadOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto derivedOp = builder.create<LoadOp>(getLoc(), derivatives.lookup(getArray()), getIndices());
    return derivedOp->getResults();
  }

  void LoadOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getArray());
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
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getOperand(), mlir::SideEffects::DefaultResource::get());
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
    mlir::Type newResultType = getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<LogOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange LogOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[ln(x)] = x' / x

    mlir::Value derivedOperand = derivatives.lookup(getOperand());

    auto derivedOp = builder.create<DivEWOp>(
        getLoc(), convertToRealType(getResult().getType()), derivedOperand, getOperand());

    return derivedOp->getResults();
  }

  void LogOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
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
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getOperand(), mlir::SideEffects::DefaultResource::get());
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
    mlir::Type newResultType = getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), getOperand(), indexes);

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
    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value ten = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 10));
    mlir::Value log = builder.create<LogOp>(loc, RealType::get(getContext()), ten);
    mlir::Value mul = builder.create<MulEWOp>(loc, type, getOperand(), log);
    auto derivedOp = builder.create<DivEWOp>(loc, getResult().getType(), derivedOperand, mul);

    return derivedOp->getResults();
  }

  void Log10Op::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
  }

  void Log10Op::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // MaxOp
  //===----------------------------------------------------------------------===//

  mlir::ParseResult MaxOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::OpAsmParser::UnresolvedOperand first;
    mlir::Type firstType;

    mlir::OpAsmParser::UnresolvedOperand second;
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

  void MaxOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << getFirst();

    if (getOperation()->getNumOperands() == 2) {
      printer << ", " << getSecond();
    }

    printer.printOptionalAttrDict(getOperation()->getAttrs());
    printer << " : ";

    if (getOperation()->getNumOperands() == 1) {
      printer << getFirst().getType();
    } else {
      printer << "(" << getFirst().getType() << ", " << getSecond().getType() << ")";
    }

    printer << " -> " << getResult().getType();
  }

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
    if (getFirst().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getFirst(), mlir::SideEffects::DefaultResource::get());
    }
  }

  //===----------------------------------------------------------------------===//
  // MinOp
  //===----------------------------------------------------------------------===//

  mlir::ParseResult MinOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::OpAsmParser::UnresolvedOperand first;
    mlir::Type firstType;

    mlir::OpAsmParser::UnresolvedOperand second;
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

  void MinOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << getFirst();

    if (getOperation()->getNumOperands() == 2) {
      printer << ", " << getSecond();
    }

    printer.printOptionalAttrDict(getOperation()->getAttrs());
    printer << " : ";

    if (getOperation()->getNumOperands() == 1) {
      printer << getFirst().getType();
    } else {
      printer << "(" << getFirst().getType() << ", " << getSecond().getType() << ")";
    }

    printer << " -> " << getResult().getType();
  }

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
    if (getFirst().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getFirst(), mlir::SideEffects::DefaultResource::get());
    }
  }

  //===----------------------------------------------------------------------===//
  // ModOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult ModOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto x = operands[0];
    auto y = operands[1];

    if (!x || !y) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(x) && isScalar(y)) {
      if (isScalarIntegerLike(x) && isScalarIntegerLike(y)) {
        auto xValue = getScalarIntegerLikeValue(x);
        auto yValue = getScalarIntegerLikeValue(y);
        return getAttr(resultType, xValue - std::floor(static_cast<double>(xValue) / yValue) * yValue);
      }

      if (isScalarFloatLike(x) && isScalarFloatLike(y)) {
        auto xValue = getScalarFloatLikeValue(x);
        auto yValue = getScalarFloatLikeValue(y);
        return getAttr(resultType, xValue - std::floor(xValue / yValue) * yValue);
      }

      if (isScalarIntegerLike(x) && isScalarFloatLike(y)) {
        auto xValue = getScalarIntegerLikeValue(x);
        auto yValue = getScalarFloatLikeValue(y);
        return getAttr(resultType, xValue - std::floor(xValue / yValue) * yValue);
      }

      if (isScalarFloatLike(x) && isScalarIntegerLike(y)) {
        auto xValue = getScalarFloatLikeValue(x);
        auto yValue = getScalarIntegerLikeValue(y);
        return getAttr(resultType, xValue - std::floor(xValue / yValue) * yValue);
      }
    }

    return {};
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
    if (getLhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getLhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (getRhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getRhs(), mlir::SideEffects::DefaultResource::get());
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
      auto right = builder.create<DivOp>(getLoc(), getLhs().getType(), nestedOperand, getRhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
          use.set(right.getResult());
      }

      replaceAllUsesWith(getLhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<DivOp>(getLoc(), getRhs().getType(), nestedOperand, getLhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
          use.set(right.getResult());
      }

      getResult().replaceAllUsesWith(getRhs());
      erase();

      return mlir::success();
    }

    return emitError("Index out of bounds: " + std::to_string(argumentIndex));
  }

  mlir::Value MulOp::distribute(mlir::OpBuilder& builder)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(*this);

    auto lhsDefiningOp = getLhs().getDefiningOp();
    auto rhsDefiningOp = getRhs().getDefiningOp();

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
      toDistribute = getRhs();
    } else {
      assert(rhsDefiningOp != nullptr);
      childOp = mlir::cast<MulOpDistributionInterface>(rhsDefiningOp);
      toDistribute = getLhs();
    }

    assert(childOp != nullptr && toDistribute != nullptr);
    return childOp.distributeMulOp(builder, getResult().getType(), toDistribute);
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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = this->getRhs();

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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = this->getRhs();

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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = this->getRhs();

    return builder.create<MulOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange MulOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(getLhs());
    mlir::Value derivedRhs = derivatives.lookup(getRhs());

    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value firstMul = builder.create<MulOp>(loc, type, derivedLhs, getRhs());
    mlir::Value secondMul = builder.create<MulOp>(loc, type, getLhs(), derivedRhs);
    auto derivedOp = builder.create<AddOp>(loc, type, firstMul, secondMul);

    return derivedOp->getResults();
  }

  void MulOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getLhs());
    toBeDerived.push_back(getRhs());
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
    if (getLhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getLhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (getRhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getRhs(), mlir::SideEffects::DefaultResource::get());
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
      auto right = builder.create<DivEWOp>(getLoc(), getLhs().getType(), nestedOperand, getRhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(getLhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<DivEWOp>(getLoc(), getRhs().getType(), nestedOperand, getLhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      getResult().replaceAllUsesWith(getRhs());
      erase();

      return mlir::success();
    }

    return emitError("Index out of bounds: " + std::to_string(argumentIndex));
  }

  mlir::Value MulEWOp::distribute(mlir::OpBuilder& builder)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(*this);

    auto lhsDefiningOp = getLhs().getDefiningOp();
    auto rhsDefiningOp = getRhs().getDefiningOp();

    if (!lhsDefiningOp && !rhsDefiningOp) {
      return getResult();
    }

    if (!mlir::isa<MulOpDistributionInterface>(getLhs().getDefiningOp()) && !mlir::isa<MulOpDistributionInterface>(getRhs().getDefiningOp())) {
      // The operation can't be propagated because none of the children
      // know how to distribute the multiplication to their children.
      return getResult();
    }

    MulOpDistributionInterface childOp;
    mlir::Value toDistribute;

    if (lhsDefiningOp != nullptr && mlir::isa<MulOpDistributionInterface>(lhsDefiningOp)) {
      childOp = mlir::cast<MulOpDistributionInterface>(lhsDefiningOp);
      toDistribute = getRhs();
    } else {
      assert(rhsDefiningOp != nullptr);
      childOp = mlir::cast<MulOpDistributionInterface>(rhsDefiningOp);
      toDistribute = getLhs();
    }

    assert(childOp != nullptr && toDistribute != nullptr);
    return childOp.distributeMulOp(builder, getResult().getType(), toDistribute);
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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = this->getRhs();

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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = this->getRhs();

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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = this->getRhs();

    return builder.create<MulEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange MulEWOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(getLhs());
    mlir::Value derivedRhs = derivatives.lookup(getRhs());

    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value firstMul = builder.create<MulEWOp>(loc, type, derivedLhs, getRhs());
    mlir::Value secondMul = builder.create<MulEWOp>(loc, type, getLhs(), derivedRhs);
    auto derivedOp = builder.create<AddEWOp>(loc, type, firstMul, secondMul);

    return derivedOp->getResults();
  }

  void MulEWOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getLhs());
    toBeDerived.push_back(getRhs());
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
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getOperand(), mlir::SideEffects::DefaultResource::get());
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
    auto right = builder.create<NegateOp>(getLoc(), getOperand().getType(), nestedOperand);

    for (auto& use : toNest.getUses()) {
      if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
        use.set(right.getResult());
      }
    }

    replaceAllUsesWith(getOperand());
    erase();

    return mlir::success();
  }

  mlir::Value NegateOp::distribute(mlir::OpBuilder& builder)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(*this);

    auto operandDefiningOp = getOperand().getDefiningOp();

    if (!operandDefiningOp) {
      return getResult();
    }

    if (auto childOp = mlir::dyn_cast<NegateOpDistributionInterface>(getOperand().getDefiningOp())) {
      return childOp.distributeNegateOp(builder, getResult().getType());
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

    mlir::Value operand = distributeFn(this->getOperand());

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

    mlir::Value operand = distributeFn(this->getOperand());

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

    mlir::Value operand = distributeFn(this->getOperand());

    return builder.create<NegateOp>(getLoc(), resultType, operand);
  }

  mlir::ValueRange NegateOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    auto derivedOp = builder.create<NegateOp>(getLoc(), convertToRealType(getResult().getType()), derivedOperand);
    return derivedOp->getResults();
  }

  void NegateOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
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
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getOperand(), mlir::SideEffects::DefaultResource::get());
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
    if (getLhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getLhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (getRhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getRhs(), mlir::SideEffects::DefaultResource::get());
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
    if (getBase().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getBase(), mlir::SideEffects::DefaultResource::get());
    }

    if (getExponent().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getExponent(), mlir::SideEffects::DefaultResource::get());
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

    mlir::Value derivedBase = derivatives.lookup(getBase());
    mlir::Value derivedExponent = derivatives.lookup(getExponent());

    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value pow = builder.create<PowOp>(loc, type, getBase(), getExponent());
    mlir::Value ln = builder.create<LogOp>(loc, type, getBase());
    mlir::Value firstOperand = builder.create<MulOp>(loc, type, derivedExponent, ln);
    mlir::Value numerator = builder.create<MulOp>(loc, type, getExponent(), derivedBase);
    mlir::Value secondOperand = builder.create<DivOp>(loc, type, numerator, getBase());
    mlir::Value sum = builder.create<AddOp>(loc, type, firstOperand, secondOperand);
    auto derivedOp = builder.create<MulOp>(loc, type, pow, sum);

    return derivedOp->getResults();
  }

  void PowOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getBase());
    toBeDerived.push_back(getExponent());
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
    if (getBase().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getBase(), mlir::SideEffects::DefaultResource::get());
    }

    if (getExponent().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getExponent(), mlir::SideEffects::DefaultResource::get());
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

    mlir::Value derivedBase = derivatives.lookup(getBase());
    mlir::Value derivedExponent = derivatives.lookup(getExponent());

    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value pow = builder.create<PowEWOp>(loc, type, getBase(), getExponent());
    mlir::Value ln = builder.create<LogOp>(loc, type, getBase());
    mlir::Value firstOperand = builder.create<MulEWOp>(loc, type, derivedExponent, ln);
    mlir::Value numerator = builder.create<MulEWOp>(loc, type, getExponent(), derivedBase);
    mlir::Value secondOperand = builder.create<DivEWOp>(loc, type, numerator, getBase());
    mlir::Value sum = builder.create<AddEWOp>(loc, type, firstOperand, secondOperand);
    auto derivedOp = builder.create<MulEWOp>(loc, type, pow, sum);

    return derivedOp->getResults();
  }

  void PowEWOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getBase());
    toBeDerived.push_back(getExponent());
  }

  void PowEWOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // RemOp
  //===----------------------------------------------------------------------===//

  mlir::OpFoldResult RemOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
  {
    auto x = operands[0];
    auto y = operands[1];

    if (!x || !y) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(x) && isScalar(y)) {
      if (isScalarIntegerLike(x) && isScalarIntegerLike(y)) {
        auto xValue = getScalarIntegerLikeValue(x);
        auto yValue = getScalarIntegerLikeValue(y);
        return getAttr(resultType, xValue % yValue);
      }

      if (isScalarFloatLike(x) && isScalarFloatLike(y)) {
        auto xValue = getScalarFloatLikeValue(x);
        auto yValue = getScalarFloatLikeValue(y);
        return getAttr(resultType, std::fmod(xValue, yValue));
      }

      if (isScalarIntegerLike(x) && isScalarFloatLike(y)) {
        auto xValue = getScalarIntegerLikeValue(x);
        auto yValue = getScalarFloatLikeValue(y);
        return getAttr(resultType, std::fmod(xValue, yValue));
      }

      if (isScalarFloatLike(x) && isScalarIntegerLike(y)) {
        auto xValue = getScalarFloatLikeValue(x);
        auto yValue = getScalarIntegerLikeValue(y);
        return getAttr(resultType, std::fmod(xValue, yValue));
      }
    }

    return {};
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
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getOperand(), mlir::SideEffects::DefaultResource::get());
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
    mlir::Type newResultType = getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
      newResultType = arrayType.getElementType();

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), getOperand(), indexes);

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
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getOperand(), mlir::SideEffects::DefaultResource::get());
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
    mlir::Type newResultType = getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
      newResultType = arrayType.getElementType();

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);

    auto op = builder.create<SinOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange SinOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[sin(x)] = x' * cos(x)

    auto loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value cos = builder.create<CosOp>(loc, type, getOperand());
    auto derivedOp = builder.create<MulEWOp>(loc, type, cos, derivedOperand);

    return derivedOp->getResults();
  }

  void SinOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
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
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getOperand(), mlir::SideEffects::DefaultResource::get());
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
    mlir::Type newResultType = getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), getOperand(), indexes);

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
    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value cosh = builder.create<CoshOp>(loc, type, getOperand());
    auto derivedOp = builder.create<MulEWOp>(loc, type, cosh, derivedOperand);

    return derivedOp->getResults();
  }

  void SinhOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
  }

  void SinhOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // SizeOp
  //===----------------------------------------------------------------------===//

  mlir::ParseResult SizeOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::OpAsmParser::UnresolvedOperand array;
    mlir::Type arrayType;

    mlir::OpAsmParser::UnresolvedOperand dimension;
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

  void SizeOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << getArray();

    if (getOperation()->getNumOperands() == 2) {
      printer << ", " << getDimension();
    }

    printer.printOptionalAttrDict(getOperation()->getAttrs());
    printer << " : ";

    if (getOperation()->getNumOperands() == 1) {
      printer << getArray().getType();
    } else {
      printer << "(" << getArray().getType() << ", " << getDimension().getType() << ")";
    }

    printer << " -> " << getResult().getType();
  }

  void SizeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), getArray(), mlir::SideEffects::DefaultResource::get());

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
    mlir::Type newResultType = getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<SqrtOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  //===----------------------------------------------------------------------===//
  // StoreOp
  //===----------------------------------------------------------------------===//

  mlir::LogicalResult StoreOp::verify()
  {
    auto indicesAmount = getIndices().size();
    auto rank = getArrayType().getRank();

    if (indicesAmount != static_cast<size_t>(rank)) {
      return emitOpError(
          "incorrect number of indices for store (expected " +
          std::to_string(rank) + ", got " + std::to_string(indicesAmount) + ")");
    }

    return mlir::success();
  }

  void StoreOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getArray(), mlir::SideEffects::DefaultResource::get());
  }

  mlir::ValueRange StoreOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto derivedOp = builder.create<StoreOp>(
        getLoc(), derivatives.lookup(getValue()), derivatives.lookup(getArray()), getIndices());

    return derivedOp->getResults();
  }

  void StoreOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getArray());
    toBeDerived.push_back(getValue());
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
    if (getLhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getLhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (getRhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getRhs(), mlir::SideEffects::DefaultResource::get());
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
      auto right = builder.create<AddOp>(getLoc(), getLhs().getType(), nestedOperand, getRhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(getLhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<SubOp>(getLoc(), getRhs().getType(), getLhs(), nestedOperand);

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(getRhs());
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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = distributeFn(this->getRhs());

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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = distributeFn(this->getRhs());

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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = distributeFn(this->getRhs());

    return builder.create<SubOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange SubOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(getLhs());
    mlir::Value derivedRhs = derivatives.lookup(getRhs());

    auto derivedOp = builder.create<SubOp>(
        loc, convertToRealType(getResult().getType()), derivedLhs, derivedRhs);

    return derivedOp->getResults();
  }

  void SubOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getLhs());
    toBeDerived.push_back(getRhs());
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
    if (getLhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getLhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (getRhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getRhs(), mlir::SideEffects::DefaultResource::get());
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
      auto right = builder.create<AddEWOp>(getLoc(), getLhs().getType(), nestedOperand, getRhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(getLhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<SubEWOp>(getLoc(), getRhs().getType(), getLhs(), nestedOperand);

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(getRhs());
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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = distributeFn(this->getRhs());

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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = distributeFn(this->getRhs());

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

    mlir::Value lhs = distributeFn(this->getLhs());
    mlir::Value rhs = distributeFn(this->getRhs());

    return builder.create<SubEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange SubEWOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(getLhs());
    mlir::Value derivedRhs = derivatives.lookup(getRhs());

    auto derivedOp = builder.create<SubEWOp>(
        loc, convertToRealType(getResult().getType()), derivedLhs, derivedRhs);

    return derivedOp->getResults();
  }

  void SubEWOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getLhs());
    toBeDerived.push_back(getRhs());
  }

  void SubEWOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // SubscriptionOp
  //===----------------------------------------------------------------------===//

  mlir::ParseResult SubscriptionOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto loc = parser.getCurrentLocation();
    mlir::OpAsmParser::UnresolvedOperand source;
    mlir::Type sourceType;
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 3> indices;
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

  void SubscriptionOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " " << getSource() << "[" << getIndices() << "]";
    printer.printOptionalAttrDict(getOperation()->getAttrs());
    printer << " : " << getSource().getType();
  }

  mlir::LogicalResult SubscriptionOp::verify()
  {
    auto indicesAmount = getIndices().size();

    if (getSourceArrayType().slice(indicesAmount) != getResultArrayType()) {
      return emitOpError("incompatible source array type and result sliced type");
    }

    return mlir::success();
  }

  mlir::ValueRange SubscriptionOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto derivedOp = builder.create<SubscriptionOp>(getLoc(), derivatives.lookup(getSource()), getIndices());
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
    effects.emplace_back(mlir::MemoryEffects::Read::get(), getMatrix(), mlir::SideEffects::DefaultResource::get());
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
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getOperand(), mlir::SideEffects::DefaultResource::get());
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
    mlir::Type newResultType = getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), getOperand(), indexes);

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
    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value cos = builder.create<CosOp>(loc, type, getOperand());
    mlir::Value two = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 2));
    mlir::Value denominator = builder.create<PowEWOp>(loc, type, cos, two);
    auto derivedOp = builder.create<DivEWOp>(loc, type, derivedOperand, denominator);

    return derivedOp->getResults();
  }

  void TanOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
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
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getOperand(), mlir::SideEffects::DefaultResource::get());
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
    mlir::Type newResultType = getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), getOperand(), indexes);

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
    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value cosh = builder.create<CoshOp>(loc, type, getOperand());
    mlir::Value two = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 2));
    mlir::Value pow = builder.create<PowEWOp>(loc, type, cosh, two);
    auto derivedOp = builder.create<DivEWOp>(loc, type, derivedOperand, pow);

    return derivedOp->getResults();
  }

  void TanhOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
  }

  void TanhOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // TransposeOp
  //===----------------------------------------------------------------------===//

  void TransposeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (getMatrix().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getMatrix(), mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  //===----------------------------------------------------------------------===//
  // WhileOp
  //===----------------------------------------------------------------------===//

  mlir::ParseResult WhileOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

  void WhileOp::print(mlir::OpAsmPrinter& printer)
  {
    printer.printRegion(getConditionRegion(), false);
    printer << " do";
    printer.printRegion(getBodyRegion(), false);
    printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  }

  mlir::ValueRange WhileOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    return llvm::None;
  }

  void WhileOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void WhileOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    regions.push_back(&getBodyRegion());
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
    if (getArray().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getArray(), mlir::SideEffects::DefaultResource::get());
    }
  }

  //===----------------------------------------------------------------------===//
  // PrintOp
  //===----------------------------------------------------------------------===//

  void PrintOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    // Ensure the operation doesn't get erased, no matter what.
    effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::SideEffects::DefaultResource::get());

    if (getValue().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), getValue(), mlir::SideEffects::DefaultResource::get());
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
