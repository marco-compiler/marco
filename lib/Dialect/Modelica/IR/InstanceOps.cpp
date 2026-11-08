#include "marco/Dialect/Modelica/IR/OpUtils.h"
#include "marco/Dialect/Modelica/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace ::mlir::modelica;
using namespace ::mlir::modelica::detail;

namespace {
/// Parse an optional keyword among the cases of an enumeration, storing the
/// corresponding attribute.
template <typename EnumAttr, typename Enum>
void parseOptionalEnumKeyword(mlir::OpAsmParser &parser,
                              mlir::OperationState &result,
                              llvm::StringRef attrName,
                              llvm::ArrayRef<Enum> cases,
                              llvm::StringRef (*stringify)(Enum)) {
  for (Enum value : cases) {
    if (mlir::succeeded(parser.parseOptionalKeyword(stringify(value)))) {
      result.addAttribute(attrName,
                          EnumAttr::get(parser.getContext(), value));
      return;
    }
  }
}

/// Get whether the type is the type of a primitive variable.
bool isVariableType(mlir::Type type) {
  if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    type = tensorType.getElementType();
  }

  return isSimpleType(type);
}
} // namespace

//===---------------------------------------------------------------------===//
// InstanceOp
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
void InstanceOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       llvm::StringRef name, mlir::Type type, ClassKind kind) {
  state.addAttribute(getSymNameAttrName(state.name),
                     builder.getStringAttr(name));

  state.addAttribute(getTypeAttrName(state.name), mlir::TypeAttr::get(type));

  state.addAttribute(getKindAttrName(state.name),
                     ClassKindAttr::get(builder.getContext(), kind));

  state.addRegion()->emplaceBlock();
}

mlir::ParseResult InstanceOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  if (mlir::succeeded(parser.parseOptionalKeyword("protected"))) {
    result.addAttribute(getIsProtectedAttrName(result.name),
                        parser.getBuilder().getUnitAttr());
  }

  llvm::StringRef kindKeyword;
  llvm::SMLoc kindLoc = parser.getCurrentLocation();

  if (parser.parseKeyword(&kindKeyword)) {
    return mlir::failure();
  }

  std::optional<ClassKind> kind = symbolizeClassKind(kindKeyword);

  if (!kind) {
    return parser.emitError(kindLoc, "unknown class kind '")
           << kindKeyword << "'";
  }

  result.addAttribute(getKindAttrName(result.name),
                      ClassKindAttr::get(parser.getContext(), *kind));

  mlir::StringAttr name;
  mlir::Type type;

  if (parser.parseSymbolName(name, getSymNameAttrName(result.name),
                             result.attributes) ||
      parser.parseColon() || parser.parseType(type)) {
    return mlir::failure();
  }

  result.addAttribute(getTypeAttrName(result.name), mlir::TypeAttr::get(type));

  if (parseRegionWithBlock(parser, *result.addRegion())) {
    return mlir::failure();
  }

  return parser.parseOptionalAttrDictWithKeyword(result.attributes);
}

void InstanceOp::print(mlir::OpAsmPrinter &printer) {
  if (getIsProtected()) {
    printer << " protected";
  }

  printer << " " << stringifyClassKind(getKind()) << " ";
  printer.printSymbolName(getSymName());
  printer << " : " << getType() << " ";
  printer.printRegion(getBodyRegion(), false);

  printAttributes(printer, getOperation(),
                  {getSymNameAttrName(), getTypeAttrName(), getKindAttrName(),
                   getIsProtectedAttrName()});
}

mlir::LogicalResult InstanceOp::verify() {
  mlir::Type type = getType();

  if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    if (!tensorType.hasStaticShape()) {
      return emitOpError("arrays of instances must have a static shape");
    }

    type = tensorType.getElementType();
  }

  if (!mlir::isa<ObjectType>(type)) {
    return emitOpError("invalid instance type ") << getType();
  }

  return mlir::success();
}

llvm::ArrayRef<int64_t> InstanceOp::getShape() {
  return modelica::getShape(getType());
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// VariableOp
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
void VariableOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       llvm::StringRef name, mlir::Type type) {
  state.addAttribute(getSymNameAttrName(state.name),
                     builder.getStringAttr(name));

  state.addAttribute(getTypeAttrName(state.name), mlir::TypeAttr::get(type));
  state.addRegion();
  state.addRegion();
}

mlir::ParseResult VariableOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  if (mlir::succeeded(parser.parseOptionalKeyword("protected"))) {
    result.addAttribute(getIsProtectedAttrName(result.name),
                        parser.getBuilder().getUnitAttr());
  }

  parseOptionalEnumKeyword<ConnectorKindAttr>(
      parser, result, getConnectorKindAttrName(result.name),
      {ConnectorKind::flow, ConnectorKind::stream}, stringifyConnectorKind);

  parseOptionalEnumKeyword<VariabilityAttr>(
      parser, result, getVariabilityAttrName(result.name),
      {Variability::discrete, Variability::parameter, Variability::constant},
      stringifyVariability);

  parseOptionalEnumKeyword<CausalityAttr>(
      parser, result, getCausalityAttrName(result.name),
      {Causality::input, Causality::output}, stringifyCausality);

  mlir::StringAttr name;
  mlir::Type type;

  if (parser.parseSymbolName(name, getSymNameAttrName(result.name),
                             result.attributes) ||
      parser.parseColon() || parser.parseType(type)) {
    return mlir::failure();
  }

  result.addAttribute(getTypeAttrName(result.name), mlir::TypeAttr::get(type));

  mlir::Region *modificationRegion = result.addRegion();
  mlir::Region *dimensionsRegion = result.addRegion();

  if (parseOptionalRegion(parser, *modificationRegion)) {
    return mlir::failure();
  }

  if (mlir::succeeded(parser.parseOptionalKeyword("dims"))) {
    if (parser.parseRegion(*dimensionsRegion)) {
      return mlir::failure();
    }
  }

  return parser.parseOptionalAttrDictWithKeyword(result.attributes);
}

void VariableOp::print(mlir::OpAsmPrinter &printer) {
  if (getIsProtected()) {
    printer << " protected";
  }

  if (auto connectorKind = getConnectorKind()) {
    printer << " " << stringifyConnectorKind(*connectorKind);
  }

  if (auto variability = getVariability()) {
    printer << " " << stringifyVariability(*variability);
  }

  if (auto causality = getCausality()) {
    printer << " " << stringifyCausality(*causality);
  }

  printer << " ";
  printer.printSymbolName(getSymName());
  printer << " : " << getType();

  if (!getModificationRegion().empty()) {
    printer << " ";

    // The block arguments are the indices of the enclosing instances.
    printer.printRegion(getModificationRegion(),
                        getModificationRegion().getNumArguments() != 0);
  }

  printOptionalRegion(printer, getDimensionsRegion(), "dims");

  printAttributes(printer, getOperation(),
                  {getSymNameAttrName(), getTypeAttrName(),
                   getConnectorKindAttrName(), getVariabilityAttrName(),
                   getCausalityAttrName(), getIsProtectedAttrName()});
}

mlir::LogicalResult VariableOp::verify() {
  mlir::Type type = getType();

  if (!isVariableType(type)) {
    return emitOpError("invalid variable type ") << type;
  }

  auto shape = modelica::getShape(type);
  size_t numDynamicDims = llvm::count_if(shape, mlir::ShapedType::isDynamic);

  if (numDynamicDims != 0 && !mlir::isa<FunctionOp>(getOperation()->getParentOp())) {
    return emitOpError("only the variables of functions can have dynamic "
                       "dimensions");
  }

  if (!getDimensionsRegion().empty()) {
    auto values = getYieldedValues(getOperation(), getDimensionsRegion(),
                                   "dimensions", numDynamicDims,
                                   numDynamicDims);

    if (mlir::failed(values)) {
      return mlir::failure();
    }

    for (mlir::Value value : *values) {
      if (!mlir::isa<IntegerType>(value.getType())) {
        return emitOpError("dimensions must be Integer scalars, but got ")
               << value.getType();
      }
    }
  }

  mlir::Region &modificationRegion = getModificationRegion();

  if (!modificationRegion.empty()) {
    if (modificationRegion.getNumArguments() != getNumEnclosingDimensions()) {
      return emitOpError("the modification region must have one argument for "
                         "each dimension of the enclosing instances");
    }

    for (mlir::BlockArgument argument : modificationRegion.getArguments()) {
      if (!mlir::isa<IntegerType>(argument.getType())) {
        return emitOpError("the arguments of the modification region must be "
                           "Integer scalars");
      }
    }
  }

  return verifyModificationRegion(getOperation(), modificationRegion);
}

size_t VariableOp::getNumEnclosingDimensions() {
  size_t result = 0;

  for (auto parentOp = getOperation()->getParentOfType<InstanceOp>(); parentOp;
       parentOp = parentOp->getParentOfType<InstanceOp>()) {
    result += parentOp.getShape().size();
  }

  return result;
}

mlir::Value VariableOp::getBinding() {
  if (getModificationRegion().empty()) {
    return nullptr;
  }

  for (auto bindOp : getModificationRegion().front().getOps<BindOp>()) {
    return bindOp.getValue();
  }

  return nullptr;
}

ModifyOp VariableOp::getAttributeModification(llvm::StringRef name) {
  if (getModificationRegion().empty()) {
    return nullptr;
  }

  for (auto modifyOp : getModificationRegion().front().getOps<ModifyOp>()) {
    if (modifyOp.getName().getNestedReferences().empty() &&
        modifyOp.getName().getRootReference() == name) {
      return modifyOp;
    }
  }

  return nullptr;
}
} // namespace mlir::modelica
