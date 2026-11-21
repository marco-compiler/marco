#include "marco/Dialect/Modelica/IR/OpUtils.h"
#include "marco/Dialect/Modelica/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace ::mlir::modelica;
using namespace ::mlir::modelica::detail;

//===---------------------------------------------------------------------===//
// Utilities
//===---------------------------------------------------------------------===//

namespace {
/// Optional keyword corresponding to a unit attribute.
struct Flag {
  llvm::StringLiteral keyword;
  llvm::StringLiteral attrName;
};

/// Prefixes of element declarations, in the order of the grammar.
constexpr Flag kElementPrefixes[] = {
    {"protected", "isProtected"}, {"redeclare", "isRedeclare"},
    {"each", "isEach"},           {"final", "isFinal"},
    {"inner", "isInner"},         {"outer", "isOuter"},
    {"replaceable", "isReplaceable"}};

void parseFlags(mlir::OpAsmParser &parser, mlir::OperationState &result,
                llvm::ArrayRef<Flag> flags) {
  for (const Flag &flag : flags) {
    if (mlir::succeeded(parser.parseOptionalKeyword(flag.keyword))) {
      result.addAttribute(flag.attrName, parser.getBuilder().getUnitAttr());
    }
  }
}

void printFlags(mlir::OpAsmPrinter &printer, mlir::Operation *op,
                llvm::ArrayRef<Flag> flags,
                llvm::SmallVectorImpl<llvm::StringRef> &elidedAttrs) {
  for (const Flag &flag : flags) {
    elidedAttrs.push_back(flag.attrName);

    if (op->getAttr(flag.attrName)) {
      printer << " " << flag.keyword;
    }
  }
}

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

mlir::ParseResult parseDimensions(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
  llvm::SmallVector<mlir::Attribute> dimensions;

  auto parseDimension = [&]() -> mlir::ParseResult {
    mlir::Attribute dimension = DimensionAttr::parse(parser, nullptr);

    if (!dimension) {
      return mlir::failure();
    }

    dimensions.push_back(dimension);
    return mlir::success();
  };

  if (parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Square,
                                     parseDimension)) {
    return mlir::failure();
  }

  result.addAttribute("dimensions",
                      parser.getBuilder().getArrayAttr(dimensions));

  return mlir::success();
}

void printDimensions(mlir::OpAsmPrinter &printer, mlir::ArrayAttr dimensions) {
  printer << "[";

  llvm::interleaveComma(dimensions, printer, [&](mlir::Attribute attr) {
    auto dimension = mlir::cast<DimensionAttr>(attr);

    switch (dimension.getKind()) {
    case DimensionKind::colon:
      printer << ":";
      break;

    case DimensionKind::fixed:
      printer << dimension.getSize();
      break;

    case DimensionKind::expression:
      printer << "?";
      break;

    case DimensionKind::boolean:
      printer << "Boolean";
      break;

    case DimensionKind::enumeration:
      printer.printAttribute(dimension.getEnumeration());
      break;
    }
  });

  printer << "]";
}

/// Verify the dimensions of a declaration against the values yielded by the
/// region computing them.
mlir::LogicalResult verifyDimensions(mlir::Operation *op,
                                     mlir::ArrayAttr dimensions,
                                     mlir::Region &region) {
  size_t numExpressions = llvm::count_if(dimensions, [](mlir::Attribute attr) {
    return mlir::cast<DimensionAttr>(attr).getKind() ==
           DimensionKind::expression;
  });

  if (numExpressions == 0) {
    if (!region.empty()) {
      return op->emitOpError("dimensions region must be empty if no "
                             "dimension is given by an expression");
    }

    return mlir::success();
  }

  auto values = getYieldedValues(op, region, "dimensions", numExpressions,
                                 numExpressions);

  if (mlir::failed(values)) {
    return mlir::failure();
  }

  for (mlir::Value value : *values) {
    if (!mlir::isa<IntegerType>(value.getType())) {
      return op->emitOpError("dimensions must be Integer scalars, but got ")
             << value.getType();
    }
  }

  return mlir::success();
}

/// Get whether the region is one where elements can be modified or
/// redeclared.
bool isModificationRegion(mlir::Region *region) {
  mlir::Operation *parentOp = region->getParentOp();

  if (auto componentOp = mlir::dyn_cast<ComponentOp>(parentOp)) {
    return region == &componentOp.getModificationRegion() ||
           region == &componentOp.getConstraintRegion();
  }

  if (auto extendsOp = mlir::dyn_cast<ExtendsOp>(parentOp)) {
    return region == &extendsOp.getModificationRegion();
  }

  if (auto variableOp = mlir::dyn_cast<VariableOp>(parentOp)) {
    return region == &variableOp.getModificationRegion();
  }

  return mlir::isa<ModifyOp, ConstrainedByOp>(parentOp);
}

/// Verify the context in which a class, enumeration or component is declared.
mlir::LogicalResult verifyElementContext(mlir::Operation *op) {
  mlir::Operation *parentOp = op->getParentOp();

  auto hasFlag = [&](llvm::StringRef name) {
    return static_cast<bool>(op->getAttr(name));
  };

  if (mlir::isa<mlir::ModuleOp>(parentOp)) {
    // Top-level definitions can only be final.
    if (mlir::isa<ComponentOp>(op)) {
      return op->emitOpError("components must be declared within classes");
    }

    for (const Flag &flag : kElementPrefixes) {
      if (flag.keyword != "final" && hasFlag(flag.attrName)) {
        return op->emitOpError("top-level definitions can't be ")
               << flag.keyword;
      }
    }

    return mlir::success();
  }

  if (mlir::isa<ClassInterface>(parentOp)) {
    if (hasFlag("isEach")) {
      return op->emitOpError("'each' is allowed only within modifications");
    }

    return mlir::success();
  }

  if (!isModificationRegion(op->getParentRegion())) {
    return op->emitOpError("elements can be declared only within the "
                           "modification regions of the parent operation");
  }

  if (!hasFlag("isRedeclare") && !hasFlag("isReplaceable")) {
    return op->emitOpError("elements declared within modifications must be "
                           "redeclare or replaceable");
  }

  if (hasFlag("isProtected") || hasFlag("isInner") || hasFlag("isOuter")) {
    return op->emitOpError("elements declared within modifications can't be "
                           "protected, inner or outer");
  }

  return mlir::success();
}
} // namespace

//===---------------------------------------------------------------------===//
// Classes
//===---------------------------------------------------------------------===//

namespace {
/// Elements found within the body of a class.
struct ClassContent {
  size_t numExtends = 0;
  size_t numInheritedExtends = 0;
  size_t numConstrainedBy = 0;
  size_t numExternal = 0;
  size_t numImports = 0;
  size_t numInstanceElements = 0;
  llvm::SmallVector<mlir::Operation *> classes;
  llvm::SmallVector<ComponentOp> components;
  llvm::SmallVector<mlir::Operation *> sections;

  /// Get the number of elements other than extends-clauses and constraining
  /// clauses.
  size_t getNumOtherElements() const {
    return numExternal + numImports + numInstanceElements + classes.size() +
           components.size() + sections.size();
  }
};

/// Verify the properties shared by the operations representing the
/// definition of a class, and collect the content of its body.
mlir::LogicalResult verifyClass(ClassInterface classOp, bool isDerivative,
                                ClassContent &content) {
  mlir::Operation *op = classOp.getOperation();

  if (mlir::failed(verifyElementContext(op))) {
    return mlir::failure();
  }

  int numDefinitionForms = static_cast<int>(classOp.getIsShort()) +
                           static_cast<int>(classOp.getIsClassExtends()) +
                           static_cast<int>(isDerivative);

  if (numDefinitionForms > 1) {
    return op->emitOpError("a class can't be at the same time a short class "
                           "definition, a class-extends redeclaration or a "
                           "derivative definition");
  }

  if (classOp.getIsClassExtends() && !op->getAttr("isRedeclare")) {
    return op->emitOpError("class-extends redeclarations must be explicitly "
                           "redeclare");
  }

  for (mlir::Operation &nested : *classOp.getBody()) {
    if (auto extendsOp = mlir::dyn_cast<ExtendsOp>(nested)) {
      if (extendsOp.getBaseClass()) {
        ++content.numExtends;
      } else {
        ++content.numInheritedExtends;
      }
    } else if (mlir::isa<ConstrainedByOp>(nested)) {
      ++content.numConstrainedBy;
    } else if (mlir::isa<ExternalOp>(nested)) {
      ++content.numExternal;
    } else if (mlir::isa<ImportOp>(nested)) {
      ++content.numImports;
    } else if (mlir::isa<ClassInterface, EnumerationOp>(nested)) {
      content.classes.push_back(&nested);
    } else if (auto componentOp = mlir::dyn_cast<ComponentOp>(nested)) {
      content.components.push_back(componentOp);
    } else if (mlir::isa<EquationSectionOp, AlgorithmSectionOp>(nested)) {
      content.sections.push_back(&nested);
    } else if (mlir::isa<VariableOp, InstanceOp>(nested)) {
      ++content.numInstanceElements;
    } else {
      return nested.emitOpError("is not allowed within the body of a class");
    }
  }

  if (content.numConstrainedBy > 1) {
    return op->emitOpError("can't have more than one constraining clause");
  }

  if (content.numConstrainedBy != 0 && !classOp.getIsReplaceable()) {
    return op->emitOpError("only replaceable classes can have a constraining "
                           "clause");
  }

  if (content.numExternal != 0 && !mlir::isa<FunctionOp>(op)) {
    return op->emitOpError("only functions can have an external clause");
  }

  if (content.numExternal > 1) {
    return op->emitOpError("can't have more than one external clause");
  }

  if (classOp.getIsShort() &&
      (content.numExtends != 1 || content.numInheritedExtends != 0 ||
       content.getNumOtherElements() != 0)) {
    return op->emitOpError("short class definitions must contain exactly one "
                           "extends-clause");
  }

  // Extends-clauses without base class in other classes are reported by the
  // verifier of the extends-clause.
  if (classOp.getIsClassExtends() && content.numInheritedExtends != 1) {
    return op->emitOpError("class-extends redeclarations must contain exactly "
                           "one extends-clause without base class");
  }

  // Classes resulting from the instantiation.
  if (content.numInstanceElements != 0) {
    if (!mlir::isa<mlir::ModuleOp>(op->getParentOp())) {
      return op->emitOpError("instantiated classes must be top-level "
                             "definitions");
    }

    if (content.numExtends != 0 || content.numInheritedExtends != 0 ||
        content.numConstrainedBy != 0 || content.numImports != 0 ||
        !content.classes.empty() || !content.components.empty()) {
      return op->emitOpError("instantiated classes can contain only "
                             "variables, instances, and equation and "
                             "algorithm sections");
    }
  }

  return mlir::success();
}

/// Verify that a class has no equation and algorithm sections.
mlir::LogicalResult verifyNoSections(const ClassContent &content,
                                     llvm::StringRef restriction) {
  if (!content.sections.empty()) {
    return content.sections.front()->emitOpError()
           << "is not allowed within " << restriction;
  }

  return mlir::success();
}

/// Verify that a class has no components.
mlir::LogicalResult verifyNoComponents(const ClassContent &content,
                                       llvm::StringRef restriction) {
  if (!content.components.empty()) {
    return content.components.front()->emitOpError()
           << "is not allowed within " << restriction;
  }

  return mlir::success();
}
} // namespace

namespace mlir::modelica {
mlir::LogicalResult ClassOp::verify() {
  ClassContent content;
  return verifyClass(*this, false, content);
}

mlir::LogicalResult ModelOp::verify() {
  ClassContent content;
  return verifyClass(*this, false, content);
}

mlir::LogicalResult BlockOp::verify() {
  ClassContent content;
  return verifyClass(*this, false, content);
}

mlir::LogicalResult RecordOp::verify() {
  ClassContent content;

  if (mlir::failed(verifyClass(*this, false, content))) {
    return mlir::failure();
  }

  return verifyNoSections(content, "records");
}

mlir::LogicalResult ConnectorOp::verify() {
  ClassContent content;

  if (mlir::failed(verifyClass(*this, false, content))) {
    return mlir::failure();
  }

  return verifyNoSections(content, "connectors");
}

mlir::LogicalResult TypeOp::verify() {
  ClassContent content;

  if (mlir::failed(verifyClass(*this, false, content)) ||
      mlir::failed(verifyNoComponents(content, "types")) ||
      mlir::failed(verifyNoSections(content, "types"))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult PackageOp::verify() {
  ClassContent content;

  if (mlir::failed(verifyClass(*this, false, content)) ||
      mlir::failed(verifyNoSections(content, "packages"))) {
    return mlir::failure();
  }

  for (ComponentOp componentOp : content.components) {
    if (componentOp.getVariability() != Variability::constant) {
      return componentOp.emitOpError("packages can contain only constant "
                                     "components");
    }
  }

  return mlir::success();
}

mlir::LogicalResult FunctionOp::verify() {
  if (static_cast<bool>(getDerivativeOfAttr()) !=
      static_cast<bool>(getDerivativeArgsAttr())) {
    return emitOpError("the function and its arguments must be specified "
                       "together for derivative definitions");
  }

  ClassContent content;

  if (mlir::failed(
          verifyClass(*this, getDerivativeOfAttr() != nullptr, content))) {
    return mlir::failure();
  }

  for (mlir::Operation *sectionOp : content.sections) {
    if (mlir::isa<EquationSectionOp>(sectionOp)) {
      return sectionOp->emitOpError("is not allowed within functions");
    }

    if (mlir::cast<AlgorithmSectionOp>(sectionOp).getIsInitial()) {
      return sectionOp->emitOpError("functions can't have initial "
                                    "algorithms");
    }
  }

  if (getDerivativeOfAttr() &&
      (content.numExtends != 0 || content.getNumOtherElements() != 0)) {
    return emitOpError("derivative definitions must have an empty body");
  }

  return mlir::success();
}

mlir::LogicalResult OperatorOp::verify() {
  ClassContent content;

  if (mlir::failed(verifyClass(*this, false, content)) ||
      mlir::failed(verifyNoComponents(content, "operators")) ||
      mlir::failed(verifyNoSections(content, "operators"))) {
    return mlir::failure();
  }

  for (mlir::Operation *nested : content.classes) {
    if (!mlir::isa<FunctionOp>(nested)) {
      return nested->emitOpError("is not allowed within operators");
    }
  }

  return mlir::success();
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// EnumerationOp
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::ParseResult EnumerationOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  parseFlags(parser, result, kElementPrefixes);

  if (mlir::succeeded(parser.parseOptionalKeyword("unspecified"))) {
    result.addAttribute(getIsUnspecifiedAttrName(result.name),
                        parser.getBuilder().getUnitAttr());
  }

  mlir::StringAttr name;

  if (parser.parseSymbolName(name, getSymNameAttrName(result.name),
                             result.attributes) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
    return mlir::failure();
  }

  return parseRegionWithBlock(parser, *result.addRegion());
}

void EnumerationOp::print(mlir::OpAsmPrinter &printer) {
  llvm::SmallVector<llvm::StringRef> elidedAttrs = {
      getSymNameAttrName(), getIsUnspecifiedAttrName()};

  printFlags(printer, getOperation(), kElementPrefixes, elidedAttrs);

  if (getIsUnspecified()) {
    printer << " unspecified";
  }

  printer << " ";
  printer.printSymbolName(getSymName());
  printAttributes(printer, getOperation(), elidedAttrs);
  printer << " ";
  printer.printRegion(getBodyRegion(), false);
}

mlir::LogicalResult EnumerationOp::verify() {
  if (mlir::failed(verifyElementContext(getOperation()))) {
    return mlir::failure();
  }

  size_t numLiterals = 0;
  size_t numConstrainedBy = 0;

  for (mlir::Operation &op : *getBody()) {
    if (mlir::isa<EnumLiteralOp>(op)) {
      ++numLiterals;
    } else if (mlir::isa<ConstrainedByOp>(op)) {
      ++numConstrainedBy;
    } else {
      return op.emitOpError("is not allowed within an enumeration");
    }
  }

  if (getIsUnspecified() && numLiterals != 0) {
    return emitOpError("unspecified enumerations can't have literals");
  }

  if (numConstrainedBy > 1) {
    return emitOpError("can't have more than one constraining clause");
  }

  if (numConstrainedBy != 0 && !getIsReplaceable()) {
    return emitOpError("only replaceable enumerations can have a "
                       "constraining clause");
  }

  return mlir::success();
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// ConstrainedByOp
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::LogicalResult ConstrainedByOp::verifyRegions() {
  return verifyModificationRegion(getOperation(), getModificationRegion());
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// ComponentOp
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
void ComponentOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        llvm::StringRef name, mlir::SymbolRefAttr className,
                        mlir::Type type) {
  state.addAttribute(getSymNameAttrName(state.name),
                     builder.getStringAttr(name));

  state.addAttribute(getClassNameAttrName(state.name), className);
  state.addAttribute(getTypeAttrName(state.name), mlir::TypeAttr::get(type));

  for (int i = 0; i < 4; ++i) {
    state.addRegion();
  }
}

mlir::ParseResult ComponentOp::parse(mlir::OpAsmParser &parser,
                                     mlir::OperationState &result) {
  parseFlags(parser, result, kElementPrefixes);

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
  mlir::SymbolRefAttr className;
  mlir::Type type;

  if (parser.parseSymbolName(name, getSymNameAttrName(result.name),
                             result.attributes) ||
      parser.parseColon() ||
      parser.parseAttribute(className, getClassNameAttrName(result.name),
                            result.attributes) ||
      parser.parseArrow() || parser.parseType(type)) {
    return mlir::failure();
  }

  result.addAttribute(getTypeAttrName(result.name), mlir::TypeAttr::get(type));

  mlir::Region *dimensionsRegion = result.addRegion();
  mlir::Region *modificationRegion = result.addRegion();
  mlir::Region *conditionRegion = result.addRegion();
  mlir::Region *constraintRegion = result.addRegion();

  if (parseOptionalRegion(parser, *modificationRegion)) {
    return mlir::failure();
  }

  if (mlir::succeeded(parser.parseOptionalKeyword("dims"))) {
    if (parseDimensions(parser, result) ||
        parseOptionalRegion(parser, *dimensionsRegion)) {
      return mlir::failure();
    }
  }

  if (mlir::succeeded(parser.parseOptionalKeyword("if"))) {
    if (parser.parseRegion(*conditionRegion)) {
      return mlir::failure();
    }
  }

  if (mlir::succeeded(parser.parseOptionalKeyword("constrainedby"))) {
    mlir::SymbolRefAttr constrainingClass;

    if (parser.parseAttribute(constrainingClass,
                              getConstrainingClassAttrName(result.name),
                              result.attributes) ||
        parseOptionalRegion(parser, *constraintRegion)) {
      return mlir::failure();
    }
  }

  return parser.parseOptionalAttrDictWithKeyword(result.attributes);
}

void ComponentOp::print(mlir::OpAsmPrinter &printer) {
  llvm::SmallVector<llvm::StringRef> elidedAttrs = {
      getSymNameAttrName(),      getClassNameAttrName(),
      getTypeAttrName(),         getConnectorKindAttrName(),
      getVariabilityAttrName(),  getCausalityAttrName(),
      getDimensionsAttrName(),   getConstrainingClassAttrName()};

  printFlags(printer, getOperation(), kElementPrefixes, elidedAttrs);

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
  printer << " : " << getClassNameAttr() << " -> " << getType();

  printOptionalRegion(printer, getModificationRegion());

  if (!getDimensions().empty()) {
    printer << " dims ";
    printDimensions(printer, getDimensions());
    printOptionalRegion(printer, getDimensionsRegion());
  }

  printOptionalRegion(printer, getConditionRegion(), "if");

  if (auto constrainingClass = getConstrainingClassAttr()) {
    printer << " constrainedby " << constrainingClass;
    printOptionalRegion(printer, getConstraintRegion());
  }

  printAttributes(printer, getOperation(), elidedAttrs);
}

mlir::LogicalResult ComponentOp::verify() {
  if (mlir::failed(verifyElementContext(getOperation()))) {
    return mlir::failure();
  }

  mlir::Type type = getType();

  if (!isScalarType(type)) {
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);

    if (!tensorType || !isScalarType(tensorType.getElementType())) {
      return emitOpError("invalid component type ") << type;
    }
  }

  if (static_cast<int64_t>(getDimensions().size()) >
      static_cast<int64_t>(modelica::getShape(type).size())) {
    return emitOpError("the type ")
           << type << " has less dimensions than the declared ones";
  }

  if (mlir::failed(verifyDimensions(getOperation(), getDimensions(),
                                    getDimensionsRegion()))) {
    return mlir::failure();
  }

  if (mlir::failed(verifyModificationRegion(getOperation(),
                                            getModificationRegion())) ||
      mlir::failed(verifyModificationRegion(getOperation(),
                                            getConstraintRegion()))) {
    return mlir::failure();
  }

  if (!getConditionRegion().empty() &&
      mlir::failed(verifyCondition(getOperation(), getConditionRegion()))) {
    return mlir::failure();
  }

  if (!getConstraintRegion().empty() && !getConstrainingClassAttr()) {
    return emitOpError("the constraining modification requires a "
                       "constraining class");
  }

  if (getConstrainingClassAttr() && !getIsReplaceable()) {
    return emitOpError("only replaceable components can have a constraining "
                       "clause");
  }

  return mlir::success();
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// ExtendsOp
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::ParseResult ExtendsOp::parse(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  if (mlir::succeeded(parser.parseOptionalKeyword("protected"))) {
    result.addAttribute(getIsProtectedAttrName(result.name),
                        parser.getBuilder().getUnitAttr());
  }

  if (mlir::failed(parser.parseOptionalKeyword("inherited"))) {
    mlir::SymbolRefAttr baseClass;

    if (parser.parseAttribute(baseClass, getBaseClassAttrName(result.name),
                              result.attributes)) {
      return mlir::failure();
    }
  }

  parseOptionalEnumKeyword<CausalityAttr>(
      parser, result, getCausalityAttrName(result.name),
      {Causality::input, Causality::output}, stringifyCausality);

  mlir::Region *dimensionsRegion = result.addRegion();
  mlir::Region *modificationRegion = result.addRegion();

  if (parseOptionalRegion(parser, *modificationRegion)) {
    return mlir::failure();
  }

  if (mlir::succeeded(parser.parseOptionalKeyword("dims"))) {
    if (parseDimensions(parser, result) ||
        parseOptionalRegion(parser, *dimensionsRegion)) {
      return mlir::failure();
    }
  }

  return parser.parseOptionalAttrDictWithKeyword(result.attributes);
}

void ExtendsOp::print(mlir::OpAsmPrinter &printer) {
  llvm::SmallVector<llvm::StringRef> elidedAttrs = {
      getIsProtectedAttrName(), getBaseClassAttrName(), getCausalityAttrName(),
      getDimensionsAttrName()};

  if (getIsProtected()) {
    printer << " protected";
  }

  if (auto baseClass = getBaseClassAttr()) {
    printer << " " << baseClass;
  } else {
    printer << " inherited";
  }

  if (auto causality = getCausality()) {
    printer << " " << stringifyCausality(*causality);
  }

  printOptionalRegion(printer, getModificationRegion());

  if (!getDimensions().empty()) {
    printer << " dims ";
    printDimensions(printer, getDimensions());
    printOptionalRegion(printer, getDimensionsRegion());
  }

  printAttributes(printer, getOperation(), elidedAttrs);
}

mlir::LogicalResult ExtendsOp::verify() {
  auto classOp = mlir::cast<ClassInterface>(getOperation()->getParentOp());

  if (!getBaseClassAttr() && !classOp.getIsClassExtends()) {
    return emitOpError("the base class can be omitted only within "
                       "class-extends redeclarations");
  }

  if ((getCausality() || !getDimensions().empty()) && !classOp.getIsShort()) {
    return emitOpError("input / output prefixes and array dimensions are "
                       "allowed only within short class definitions");
  }

  if (classOp.getIsShort() && getIsProtected()) {
    return emitOpError("the extends-clause of a short class definition "
                       "can't be protected");
  }

  if (classOp.getIsShort() && !getModificationRegion().empty()) {
    // Selective model extension is not available for short class definitions.
    for (mlir::Operation &op : getModificationRegion().front()) {
      if (mlir::isa<DeselectOp, DeselectConnectOp>(op)) {
        return op.emitOpError("is not allowed within short class definitions");
      }
    }
  }

  if (mlir::failed(verifyDimensions(getOperation(), getDimensions(),
                                    getDimensionsRegion()))) {
    return mlir::failure();
  }

  return verifyModificationRegion(getOperation(), getModificationRegion());
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// ImportOp
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::ParseResult ImportOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
  if (mlir::succeeded(parser.parseOptionalKeyword("protected"))) {
    result.addAttribute(getIsProtectedAttrName(result.name),
                        parser.getBuilder().getUnitAttr());
  }

  mlir::SymbolRefAttr first;
  llvm::SMLoc firstLoc = parser.getCurrentLocation();

  if (parser.parseAttribute(first)) {
    return mlir::failure();
  }

  if (mlir::succeeded(parser.parseOptionalEqual())) {
    auto alias = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(first);

    if (!alias) {
      return parser.emitError(firstLoc, "the alias must be a simple name");
    }

    mlir::SymbolRefAttr path;

    if (parser.parseAttribute(path)) {
      return mlir::failure();
    }

    result.addAttribute(getAliasAttrName(result.name), alias);
    result.addAttribute(getPathAttrName(result.name), path);
  } else {
    result.addAttribute(getPathAttrName(result.name), first);

    if (mlir::succeeded(parser.parseOptionalKeyword("unqualified"))) {
      result.addAttribute(getIsUnqualifiedAttrName(result.name),
                          parser.getBuilder().getUnitAttr());
    } else if (mlir::succeeded(parser.parseOptionalLBrace())) {
      llvm::SmallVector<mlir::Attribute> names;

      do {
        mlir::FlatSymbolRefAttr importedName;

        if (parser.parseAttribute(importedName)) {
          return mlir::failure();
        }

        names.push_back(importedName);
      } while (mlir::succeeded(parser.parseOptionalComma()));

      if (parser.parseRBrace()) {
        return mlir::failure();
      }

      result.addAttribute(getNamesAttrName(result.name),
                          parser.getBuilder().getArrayAttr(names));
    }
  }

  return parser.parseOptionalAttrDict(result.attributes);
}

void ImportOp::print(mlir::OpAsmPrinter &printer) {
  if (getIsProtected()) {
    printer << " protected";
  }

  printer << " ";

  if (auto alias = getAliasAttr()) {
    printer << alias << " = ";
  }

  printer << getPathAttr();

  if (getIsUnqualified()) {
    printer << " unqualified";
  }

  if (auto names = getNamesAttr()) {
    printer << " {";
    llvm::interleaveComma(names, printer);
    printer << "}";
  }

  printAttributes(printer, getOperation(),
                  {getIsProtectedAttrName(), getAliasAttrName(),
                   getPathAttrName(), getIsUnqualifiedAttrName(),
                   getNamesAttrName()},
                  false);
}

mlir::LogicalResult ImportOp::verify() {
  int numForms = static_cast<int>(getAliasAttr() != nullptr) +
                 static_cast<int>(getIsUnqualified()) +
                 static_cast<int>(getNamesAttr() != nullptr);

  if (numForms > 1) {
    return emitOpError("an import-clause can be only one of renaming, "
                       "unqualified, or multiple definitions import");
  }

  if (auto names = getNamesAttr(); names && names.empty()) {
    return emitOpError("the list of imported definitions can't be empty");
  }

  return mlir::success();
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Modifications
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::LogicalResult ModifyOp::verifyRegions() {
  if (!isModificationRegion(getOperation()->getParentRegion())) {
    return emitOpError("must be within the modification regions of the parent "
                       "operation");
  }

  return verifyModificationRegion(getOperation(), getModificationRegion());
}

mlir::LogicalResult BindOp::verify() {
  if (static_cast<bool>(getValue()) == getIsBreak()) {
    return emitOpError("must have either a value or 'break'");
  }

  if (auto variableOp =
          mlir::dyn_cast<VariableOp>(getOperation()->getParentOp())) {
    if (mlir::Value value = getValue();
        value && !isCoercibleTo(value.getType(), variableOp.getType())) {
      return emitOpError("value of type ")
             << value.getType()
             << " is not compatible with the variable type "
             << variableOp.getType();
    }

    return mlir::success();
  }

  auto componentOp = mlir::dyn_cast<ComponentOp>(getOperation()->getParentOp());

  if (!componentOp) {
    return mlir::success();
  }

  if (getOperation()->getParentRegion() != &componentOp.getModificationRegion()) {
    return emitOpError("must be within the modification region of the "
                       "component");
  }

  // The value of the declaration equation must be compatible with the type of
  // the component.
  if (mlir::Value value = getValue();
      value && !isCoercibleTo(value.getType(), componentOp.getType())) {
    return emitOpError("value of type ")
           << value.getType() << " is not compatible with the component type "
           << componentOp.getType();
  }

  return mlir::success();
}

mlir::LogicalResult DeselectConnectOp::verifyRegions() {
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
// ExternalOp
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::LogicalResult ExternalOp::verify() {
  if (!getFunction()) {
    if (getHasOutput() || !getBodyRegion().empty()) {
      return emitOpError("the output and the arguments can be specified only "
                         "together with the name of the external function");
    }

    return mlir::success();
  }

  // Call without arguments and without output.
  if (getBodyRegion().empty() && !getHasOutput()) {
    return mlir::success();
  }

  size_t minValues = getHasOutput() ? 1 : 0;
  auto values = getYieldedValues(getOperation(), getBodyRegion(), "body",
                                 minValues, std::numeric_limits<size_t>::max());

  if (mlir::failed(values)) {
    return mlir::failure();
  }

  if (getHasOutput()) {
    return verifyComponentReference(getOperation(), (*values)[0], "output");
  }

  return mlir::success();
}
} // namespace mlir::modelica
