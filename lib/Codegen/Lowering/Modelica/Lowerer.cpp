#include "marco/Codegen/Lowering/Modelica/Lowerer.h"
#include "marco/Codegen/Lowering/Modelica/LowererImpl.h"
#include "llvm/ADT/ScopeExit.h"

using namespace ::marco;
using namespace ::marco::ast::modelica;
using namespace ::marco::codegen::lowering::modelica;
using namespace ::mlir::modelica;

//===---------------------------------------------------------------------===//
// Lowerer
//===---------------------------------------------------------------------===//

namespace marco::codegen::lowering::modelica {
Lowerer::Lowerer(mlir::MLIRContext &context)
    : impl(std::make_unique<LowererImpl>(context)) {}

Lowerer::~Lowerer() = default;

mlir::LogicalResult Lowerer::addFile(const StoredDefinition &file) {
  if (auto missingPackage = impl->getClassTree().add(file)) {
    return impl->emitError(impl->loc(file),
                           "the package '" + *missingPackage +
                               "' of the within-clause is not defined");
  }

  return mlir::success();
}

mlir::LogicalResult Lowerer::lower(mlir::ModuleOp module) {
  return impl->lower(module);
}
} // namespace marco::codegen::lowering::modelica

//===---------------------------------------------------------------------===//
// Utilities
//===---------------------------------------------------------------------===//

namespace {
ClassKind getClassKind(ClassRestriction restriction) {
  switch (restriction) {
  case ClassRestriction::Class:
    return ClassKind::class_;
  case ClassRestriction::Model:
    return ClassKind::model;
  case ClassRestriction::Record:
    return ClassKind::record;
  case ClassRestriction::OperatorRecord:
    return ClassKind::operator_record;
  case ClassRestriction::Block:
    return ClassKind::block;
  case ClassRestriction::Connector:
    return ClassKind::connector;
  case ClassRestriction::ExpandableConnector:
    return ClassKind::expandable_connector;
  case ClassRestriction::Type:
    return ClassKind::type;
  case ClassRestriction::Package:
    return ClassKind::package;
  case ClassRestriction::Function:
    return ClassKind::function;
  case ClassRestriction::Operator:
    return ClassKind::operator_;
  case ClassRestriction::OperatorFunction:
    return ClassKind::operator_function;
  }

  llvm_unreachable("Unknown class restriction");
  return ClassKind::class_;
}

/// Set the prefixes of an element declaration.
template <typename Op>
void setElementPrefixes(Op op, const ElementPrefixes &prefixes, bool isEach) {
  op.setIsProtected(prefixes.isProtected);
  op.setIsRedeclare(prefixes.isRedeclare);
  op.setIsEach(isEach);
  op.setIsFinal(prefixes.isFinal);
  op.setIsInner(prefixes.isInner);
  op.setIsOuter(prefixes.isOuter);
  op.setIsReplaceable(prefixes.isReplaceable);
}

/// Create an operation without operands and results, with the given number
/// of empty regions. The attributes are expected to be set afterwards.
template <typename Op>
Op createOp(mlir::OpBuilder &builder, mlir::Location location,
            unsigned numRegions = 0) {
  mlir::OperationState state(location, Op::getOperationName());

  for (unsigned i = 0; i < numRegions; ++i) {
    state.addRegion();
  }

  return mlir::cast<Op>(builder.create(state));
}
} // namespace

namespace marco::codegen::lowering::modelica {
LowererImpl::LowererImpl(mlir::MLIRContext &context) : builder(&context) {
  context.loadDialect<ModelicaDialect>();
}

mlir::Location LowererImpl::loc(const ast::ASTNode &node) {
  return loc(node.getLocation());
}

mlir::Location LowererImpl::loc(const SourceRange &range) {
  if (!range.begin.file) {
    return builder.getUnknownLoc();
  }

  return mlir::FileLineColLoc::get(
      builder.getContext(), range.begin.file->getFileName(),
      static_cast<unsigned>(range.begin.line),
      static_cast<unsigned>(range.begin.column));
}

mlir::LogicalResult LowererImpl::emitError(mlir::Location location,
                                           const llvm::Twine &message) {
  hadErrors = true;
  mlir::emitError(location) << message;
  return mlir::failure();
}

mlir::SymbolRefAttr LowererImpl::getSymbolRef(const Name &name) {
  llvm::SmallVector<mlir::FlatSymbolRefAttr> nested;
  llvm::StringRef rootName = name.global ? "." : llvm::StringRef(name.parts[0]);

  for (size_t i = name.global ? 0 : 1; i < name.parts.size(); ++i) {
    nested.push_back(mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  name.parts[i]));
  }

  return mlir::SymbolRefAttr::get(builder.getContext(), rootName, nested);
}

mlir::SymbolRefAttr LowererImpl::getAbsoluteSymbolRef(ClassNode *classNode) {
  llvm::SmallVector<llvm::StringRef> path = classNode->getPath();
  llvm::SmallVector<mlir::FlatSymbolRefAttr> nested;

  for (llvm::StringRef part : llvm::drop_begin(path)) {
    nested.push_back(mlir::FlatSymbolRefAttr::get(builder.getContext(), part));
  }

  return mlir::SymbolRefAttr::get(builder.getContext(), path.front(), nested);
}

void LowererImpl::pushIterator(llvm::StringRef name, mlir::Value value) {
  iterators.emplace_back(name.str(), value);
}

void LowererImpl::popIterator() { iterators.pop_back(); }

mlir::Value LowererImpl::lookupIterator(llvm::StringRef name) const {
  for (const auto &iterator : llvm::reverse(iterators)) {
    if (iterator.first == name) {
      return iterator.second;
    }
  }

  return nullptr;
}

//===---------------------------------------------------------------------===//
// Annotations
//===---------------------------------------------------------------------===//

namespace {
/// Convert the value of an annotation into an attribute. Expressions that
/// can't be represented by builtin attributes are converted to unit
/// attributes.
mlir::Attribute getAnnotationValue(LowererImpl &lowerer,
                                   const Expression &expression) {
  mlir::OpBuilder &builder = lowerer.builder;

  if (auto literal = expression.dyn_cast<IntegerLiteral>()) {
    return builder.getI64IntegerAttr(literal->getValue());
  }

  if (auto literal = expression.dyn_cast<RealLiteral>()) {
    return builder.getF64FloatAttr(literal->getValue());
  }

  if (auto literal = expression.dyn_cast<StringLiteral>()) {
    return builder.getStringAttr(literal->getValue());
  }

  if (auto literal = expression.dyn_cast<BooleanLiteral>()) {
    return builder.getBoolAttr(literal->getValue());
  }

  if (auto operation = expression.dyn_cast<UnaryOperation>();
      operation && operation->getOperator() == UnaryOperator::Minus) {
    if (auto literal = operation->getOperand()->dyn_cast<IntegerLiteral>()) {
      return builder.getI64IntegerAttr(-literal->getValue());
    }

    if (auto literal = operation->getOperand()->dyn_cast<RealLiteral>()) {
      return builder.getF64FloatAttr(-literal->getValue());
    }
  }

  if (auto constructor = expression.dyn_cast<ArrayConstructor>();
      constructor && constructor->getIterators().empty()) {
    llvm::SmallVector<mlir::Attribute> elements;

    for (const Child<Expression> &element : constructor->getElements()) {
      elements.push_back(getAnnotationValue(lowerer, *element));
    }

    return builder.getArrayAttr(elements);
  }

  if (auto reference = expression.dyn_cast<ComponentReference>()) {
    return lowerer.getSymbolRef(reference->getName());
  }

  if (auto call = expression.dyn_cast<Call>()) {
    mlir::NamedAttrList attrs;
    attrs.append("$call", lowerer.getSymbolRef(call->getCallee()->getName()));

    llvm::SmallVector<mlir::Attribute> args;

    for (const Child<Expression> &arg : call->getArguments()) {
      args.push_back(getAnnotationValue(lowerer, *arg));
    }

    if (!args.empty()) {
      attrs.append("$args", builder.getArrayAttr(args));
    }

    for (const Child<NamedArgument> &arg : call->getNamedArguments()) {
      if (!attrs.get(arg->getName())) {
        attrs.append(arg->getName(), getAnnotationValue(lowerer, *arg->getValue()));
      }
    }

    return attrs.getDictionary(builder.getContext());
  }

  return builder.getUnitAttr();
}
} // namespace

mlir::DictionaryAttr
LowererImpl::getAnnotation(const ClassModification *annotation) {
  if (!annotation) {
    return nullptr;
  }

  mlir::NamedAttrList attrs;

  for (const Child<Argument> &argument : annotation->getArguments()) {
    auto modification = argument->dyn_cast<ElementModification>();

    if (!modification) {
      continue;
    }

    std::string name = modification->getName().toString();

    if (attrs.get(name)) {
      continue;
    }

    const Modification *value = modification->getModification();

    if (!value) {
      attrs.append(name, builder.getUnitAttr());
      continue;
    }

    mlir::DictionaryAttr nested = getAnnotation(value->getClassModification());

    if (!value->getValue()) {
      attrs.append(name, nested ? mlir::Attribute(nested)
                                : mlir::Attribute(builder.getUnitAttr()));
      continue;
    }

    mlir::Attribute converted = getAnnotationValue(*this, *value->getValue());

    if (nested) {
      mlir::NamedAttrList merged(nested);
      merged.append("$value", converted);
      attrs.append(name, merged.getDictionary(builder.getContext()));
    } else {
      attrs.append(name, converted);
    }
  }

  return attrs.getDictionary(builder.getContext());
}

//===---------------------------------------------------------------------===//
// Classes
//===---------------------------------------------------------------------===//

mlir::LogicalResult LowererImpl::lower(mlir::ModuleOp module) {
  // Errors may also be emitted by the type inference of the operations, so
  // all the errors are observed and then forwarded to the other handlers.
  mlir::ScopedDiagnosticHandler errorObserver(
      builder.getContext(), [&](mlir::Diagnostic &diagnostic) {
        if (diagnostic.getSeverity() == mlir::DiagnosticSeverity::Error) {
          hadErrors = true;
        }

        return mlir::failure();
      });

  builder.setInsertionPointToEnd(module.getBody());

  for (const std::unique_ptr<ClassNode> &child :
       classTree.getRoot()->getChildren()) {
    if (!child->isPredefined()) {
      lowerClass(child.get());
    }
  }

  return mlir::failure(hadErrors);
}

void LowererImpl::lowerClass(ClassNode *classNode) {
  const ClassDefinition *definition = classNode->getDefinition();
  mlir::OpBuilder::InsertionGuard guard(builder);
  ClassNode *previousScope = scope;
  scope = classNode;

  // Errors are reported by the lowering of the class, which is omitted from
  // the output.
  (void)lowerClassDefinition(*definition, classNode, classNode,
                             definition->getPrefixes(), false,
                             definition->getConstrainingClause());

  scope = previousScope;
}

mlir::LogicalResult LowererImpl::lowerClassDefinition(
    const ClassDefinition &definition, ClassNode *classNode,
    ClassNode *classScope, const ElementPrefixes &prefixes, bool isEach,
    const ConstrainingClause *constraint) {
  mlir::Location location = loc(definition);
  mlir::OpBuilder::InsertionGuard guard(builder);
  ClassNode *previousScope = scope;
  scope = classScope;

  auto restoreScope = llvm::make_scope_exit([&]() { scope = previousScope; });

  auto lowerConstraint = [&]() -> mlir::LogicalResult {
    if (!constraint) {
      return mlir::success();
    }

    auto constrainedBy = createOp<ConstrainedByOp>(builder, loc(*constraint), 1);
    constrainedBy.setConstrainingClassAttr(
        getSymbolRef(constraint->getTypeSpecifier()));

    if (const ClassModification *modification =
            constraint->getClassModification()) {
      return lowerClassModification(*modification,
                                    constrainedBy.getModificationRegion());
    }

    return mlir::success();
  };

  auto setDocumentation = [&](auto op) {
    if (!definition.getDescription().text.empty()) {
      op.setDescriptionAttr(
          builder.getStringAttr(definition.getDescription().text));
    }

    if (auto annotation =
            getAnnotation(definition.getDescription().annotation.get())) {
      op.setAnnotationAttr(annotation);
    }
  };

  // Enumerations.
  if (definition.getForm() == ClassDefinition::Form::Enumeration) {
    auto enumerationOp = createOp<EnumerationOp>(builder, location, 1);
    enumerationOp.setSymNameAttr(builder.getStringAttr(definition.getName()));
    setElementPrefixes(enumerationOp, prefixes, isEach);
    enumerationOp.setIsUnspecified(definition.isUnspecifiedEnumeration());
    setDocumentation(enumerationOp);

    builder.setInsertionPointToStart(
        &enumerationOp.getBodyRegion().emplaceBlock());

    for (const Child<EnumerationLiteral> &literal :
         definition.getEnumerationLiterals()) {
      auto literalOp = createOp<EnumLiteralOp>(builder, loc(*literal));
      literalOp.setSymNameAttr(builder.getStringAttr(literal->getName()));

      if (!literal->getDescription().text.empty()) {
        literalOp.setDescriptionAttr(
            builder.getStringAttr(literal->getDescription().text));
      }
    }

    if (mlir::failed(lowerConstraint())) {
      enumerationOp.erase();
      return mlir::failure();
    }

    return mlir::success();
  }

  auto classOp = builder.create<ClassOp>(location, definition.getName(),
                                         getClassKind(definition.getRestriction()));

  setElementPrefixes(classOp, prefixes, isEach);
  classOp.setIsEncapsulated(definition.isEncapsulated());
  classOp.setIsPartial(definition.isPartial());
  classOp.setIsImpure(definition.getPurity() == Purity::Impure);
  setDocumentation(classOp);

  builder.setInsertionPointToStart(classOp.getBody());

  auto lowerBody = [&]() -> mlir::LogicalResult {
    switch (definition.getForm()) {
    case ClassDefinition::Form::Short: {
      classOp.setIsShort(true);

      auto extendsOp = createOp<ExtendsOp>(builder, location, 2);
      extendsOp.setBaseClassAttr(getSymbolRef(definition.getBaseClass()));

      if (definition.getBasePrefix() == ast::modelica::Causality::Input) {
        extendsOp.setCausality(mlir::modelica::Causality::input);
      } else if (definition.getBasePrefix() == ast::modelica::Causality::Output) {
        extendsOp.setCausality(mlir::modelica::Causality::output);
      }

      if (const ClassModification *modification =
              definition.getClassModification()) {
        if (mlir::failed(lowerClassModification(
                *modification, extendsOp.getModificationRegion()))) {
          return mlir::failure();
        }
      }

      if (!definition.getSubscripts().empty()) {
        llvm::SmallVector<const Expression *> subscripts;

        for (const Child<Expression> &subscript : definition.getSubscripts()) {
          subscripts.push_back(subscript.get());
        }

        llvm::SmallVector<mlir::Attribute> dimensions;

        if (mlir::failed(lowerDimensions(subscripts, dimensions,
                                         extendsOp.getDimensionsRegion(),
                                         location))) {
          return mlir::failure();
        }

        extendsOp.setDimensionsAttr(builder.getArrayAttr(dimensions));
      }

      return mlir::success();
    }

    case ClassDefinition::Form::Derivative: {
      classOp.setDerivativeOfAttr(
          getSymbolRef(definition.getDerivedFunction()));

      llvm::SmallVector<llvm::StringRef> args;

      for (const std::string &arg : definition.getDerivativeArguments()) {
        args.push_back(arg);
      }

      classOp.setDerivativeArgsAttr(builder.getStrArrayAttr(args));
      return mlir::success();
    }

    case ClassDefinition::Form::Enumeration:
      llvm_unreachable("Enumerations are handled separately");
      return mlir::failure();

    case ClassDefinition::Form::ClassExtends: {
      classOp.setIsClassExtends(true);
      auto extendsOp = createOp<ExtendsOp>(builder, location, 2);

      if (const ClassModification *modification =
              definition.getClassModification()) {
        if (mlir::failed(lowerClassModification(
                *modification, extendsOp.getModificationRegion()))) {
          return mlir::failure();
        }
      }

      [[fallthrough]];
    }

    case ClassDefinition::Form::Long:
      // The elements whose lowering fails are reported and omitted, so that
      // the failure doesn't affect the other elements and the nested classes.
      for (const Child<Node> &element : definition.getElements()) {
        (void)lowerElement(*element, classNode);
      }

      // Classes defined in other files through within-clauses
      // are not elements of the definition.
      if (classNode) {
        for (const std::unique_ptr<ClassNode> &child : classNode->getChildren()) {
          if (child->getDefinition()->getParent() != &definition) {
            lowerClass(child.get());
          }
        }
      }

      if (const ExternalClause *externalClause =
              definition.getExternalClause()) {
        (void)lowerExternalClause(*externalClause);
      }

      return mlir::success();
    }

    return mlir::success();
  };

  if (mlir::failed(lowerBody()) || mlir::failed(lowerConstraint())) {
    classOp.erase();
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult LowererImpl::lowerElement(const Node &element,
                                              ClassNode *classNode) {
  if (auto classDefinition = element.dyn_cast<ClassDefinition>()) {
    // Nested classes are lowered independently, so that their failure
    // doesn't affect the enclosing class.
    if (ClassNode *child = classNode->getChild(classDefinition->getName());
        child && child->getDefinition() == classDefinition) {
      lowerClass(child);
    }

    return mlir::success();
  }

  if (auto clause = element.dyn_cast<ComponentClause>()) {
    bool failed = false;

    for (const Child<ComponentDeclaration> &declaration :
         clause->getDeclarations()) {
      Element component =
          classNode->getComponents().lookup(declaration->getName());

      if (component.declaration != declaration.get()) {
        return emitError(loc(*declaration), "duplicate declaration of '" +
                                                declaration->getName() + "'");
      }

      if (mlir::failed(lowerComponent(*clause, *declaration,
                                      clause->getPrefixes(), false, component,
                                      clause->getConstrainingClause()))) {
        failed = true;
      }
    }

    return mlir::failure(failed);
  }

  if (auto extendsClause = element.dyn_cast<ExtendsClause>()) {
    return lowerExtendsClause(*extendsClause);
  }

  if (auto importClause = element.dyn_cast<ImportClause>()) {
    return lowerImportClause(*importClause);
  }

  if (auto section = element.dyn_cast<EquationSection>()) {
    return lowerEquationSection(*section);
  }

  if (auto section = element.dyn_cast<AlgorithmSection>()) {
    return lowerAlgorithmSection(*section);
  }

  return emitError(loc(element), "unexpected element");
}

//===---------------------------------------------------------------------===//
// Components
//===---------------------------------------------------------------------===//

mlir::LogicalResult LowererImpl::lowerComponent(
    const ComponentClause &clause, const ComponentDeclaration &declaration,
    const ElementPrefixes &prefixes, bool isEach, const Element &element,
    const ConstrainingClause *constraint) {
  mlir::Location location = loc(declaration);
  auto type = getComponentType(element);

  if (mlir::failed(type)) {
    return mlir::failure();
  }

  auto componentOp = builder.create<ComponentOp>(
      location, declaration.getName(), getSymbolRef(clause.getTypeSpecifier()),
      *type);

  setElementPrefixes(componentOp, prefixes, isEach);

  const TypePrefix &typePrefix = clause.getTypePrefix();

  if (typePrefix.connectorKind == ast::modelica::ConnectorKind::Flow) {
    componentOp.setConnectorKind(mlir::modelica::ConnectorKind::flow);
  } else if (typePrefix.connectorKind == ast::modelica::ConnectorKind::Stream) {
    componentOp.setConnectorKind(mlir::modelica::ConnectorKind::stream);
  }

  if (typePrefix.variability == ast::modelica::Variability::Discrete) {
    componentOp.setVariability(mlir::modelica::Variability::discrete);
  } else if (typePrefix.variability == ast::modelica::Variability::Parameter) {
    componentOp.setVariability(mlir::modelica::Variability::parameter);
  } else if (typePrefix.variability == ast::modelica::Variability::Constant) {
    componentOp.setVariability(mlir::modelica::Variability::constant);
  }

  if (typePrefix.causality == ast::modelica::Causality::Input) {
    componentOp.setCausality(mlir::modelica::Causality::input);
  } else if (typePrefix.causality == ast::modelica::Causality::Output) {
    componentOp.setCausality(mlir::modelica::Causality::output);
  }

  if (!declaration.getDescription().text.empty()) {
    componentOp.setDescriptionAttr(
        builder.getStringAttr(declaration.getDescription().text));
  }

  if (auto annotation =
          getAnnotation(declaration.getDescription().annotation.get())) {
    componentOp.setAnnotationAttr(annotation);
  }

  auto lowerRegions = [&]() -> mlir::LogicalResult {
    if (const Modification *modification = declaration.getModification()) {
      if (mlir::failed(lowerModification(
              *modification, componentOp.getModificationRegion()))) {
        return mlir::failure();
      }
    }

    // The dimensions of the declaration precede the ones of the type
    // specifier.
    llvm::SmallVector<const Expression *> subscripts;

    for (const Child<Expression> &subscript : declaration.getSubscripts()) {
      subscripts.push_back(subscript.get());
    }

    for (const Child<Expression> &subscript : clause.getSubscripts()) {
      subscripts.push_back(subscript.get());
    }

    if (!subscripts.empty()) {
      llvm::SmallVector<mlir::Attribute> dimensions;

      if (mlir::failed(lowerDimensions(subscripts, dimensions,
                                       componentOp.getDimensionsRegion(),
                                       location))) {
        return mlir::failure();
      }

      componentOp.setDimensionsAttr(builder.getArrayAttr(dimensions));
    }

    if (const Expression *condition = declaration.getCondition()) {
      if (mlir::failed(lowerIntoRegion(componentOp.getConditionRegion(),
                                       {condition}, location))) {
        return mlir::failure();
      }
    }

    if (constraint) {
      componentOp.setConstrainingClassAttr(
          getSymbolRef(constraint->getTypeSpecifier()));

      if (const ClassModification *modification =
              constraint->getClassModification()) {
        return lowerClassModification(*modification,
                                      componentOp.getConstraintRegion());
      }
    }

    return mlir::success();
  };

  if (mlir::failed(lowerRegions())) {
    componentOp.erase();
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult LowererImpl::lowerDimensions(
    llvm::ArrayRef<const Expression *> subscripts,
    llvm::SmallVectorImpl<mlir::Attribute> &dimensions, mlir::Region &region,
    mlir::Location location) {
  mlir::MLIRContext *context = builder.getContext();
  llvm::SmallVector<const Expression *> expressions;

  for (const Expression *subscript : subscripts) {
    if (subscript->isa<Colon>()) {
      dimensions.push_back(DimensionAttr::get(context, DimensionKind::colon));
      continue;
    }

    if (auto literal = subscript->dyn_cast<IntegerLiteral>()) {
      dimensions.push_back(DimensionAttr::get(context, literal->getValue()));
      continue;
    }

    // Dimensions indexed by Boolean or enumeration types.
    if (auto reference = subscript->dyn_cast<ComponentReference>()) {
      if (auto element = classTree.lookup(scope, reference->getName());
          element && element->isClass()) {
        ClassNode *classNode = element->classNode;

        if (classNode->getPredefined() == PredefinedClass::Boolean) {
          dimensions.push_back(
              DimensionAttr::get(context, DimensionKind::boolean));
          continue;
        }

        dimensions.push_back(
            DimensionAttr::get(context, getSymbolRef(reference->getName())));
        continue;
      }
    }

    dimensions.push_back(DimensionAttr::get(context, DimensionKind::expression));
    expressions.push_back(subscript);
  }

  if (expressions.empty()) {
    return mlir::success();
  }

  return lowerIntoRegion(region, expressions, location);
}

//===---------------------------------------------------------------------===//
// Extends, import and external clauses
//===---------------------------------------------------------------------===//

mlir::LogicalResult
LowererImpl::lowerExtendsClause(const ExtendsClause &clause) {
  auto extendsOp = createOp<ExtendsOp>(builder, loc(clause), 2);
  extendsOp.setBaseClassAttr(getSymbolRef(clause.getBaseClass()));
  extendsOp.setIsProtected(clause.isProtected());

  if (auto annotation = getAnnotation(clause.getAnnotation())) {
    extendsOp.setAnnotationAttr(annotation);
  }

  if (const ClassModification *modification = clause.getClassModification()) {
    if (mlir::failed(lowerClassModification(
            *modification, extendsOp.getModificationRegion()))) {
      extendsOp.erase();
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult
LowererImpl::lowerImportClause(const ImportClause &clause) {
  auto importOp = createOp<ImportOp>(builder, loc(clause));
  Name path = clause.getName();
  path.global = false;
  importOp.setPathAttr(getSymbolRef(path));
  importOp.setIsProtected(clause.isProtected());

  switch (clause.getForm()) {
  case ImportClause::Form::Qualified:
    break;

  case ImportClause::Form::Renaming:
    importOp.setAliasAttr(
        mlir::FlatSymbolRefAttr::get(builder.getContext(), clause.getAlias()));
    break;

  case ImportClause::Form::Unqualified:
    importOp.setIsUnqualified(true);
    break;

  case ImportClause::Form::Multiple: {
    llvm::SmallVector<mlir::Attribute> names;

    for (const std::string &name : clause.getImportedNames()) {
      names.push_back(mlir::FlatSymbolRefAttr::get(builder.getContext(), name));
    }

    importOp.setNamesAttr(builder.getArrayAttr(names));
    break;
  }
  }

  if (!clause.getDescription().text.empty()) {
    importOp.setDescriptionAttr(
        builder.getStringAttr(clause.getDescription().text));
  }

  return mlir::success();
}

mlir::LogicalResult
LowererImpl::lowerExternalClause(const ExternalClause &clause) {
  mlir::Location location = loc(clause);
  auto externalOp = createOp<ExternalOp>(builder, location, 1);

  // The default language is C.
  externalOp.setLanguageAttr(
      builder.getStringAttr(clause.getLanguage().value_or("C")));

  if (auto annotation = getAnnotation(clause.getAnnotation())) {
    externalOp.setAnnotationAttr(annotation);
  }

  if (!clause.hasFunctionCall()) {
    return mlir::success();
  }

  externalOp.setFunctionAttr(builder.getStringAttr(clause.getFunction()));

  llvm::SmallVector<const Expression *> values;

  if (const ComponentReference *output = clause.getOutput()) {
    externalOp.setHasOutput(true);
    values.push_back(output);
  }

  for (const Child<Expression> &argument : clause.getArguments()) {
    values.push_back(argument.get());
  }

  if (values.empty()) {
    return mlir::success();
  }

  if (mlir::failed(
          lowerIntoRegion(externalOp.getBodyRegion(), values, location))) {
    externalOp.erase();
    return mlir::failure();
  }

  return mlir::success();
}

//===---------------------------------------------------------------------===//
// Modifications
//===---------------------------------------------------------------------===//

namespace {
/// Get the block of a region, creating it if needed, and set the insertion
/// point to its end.
void setInsertionPointToEnd(mlir::OpBuilder &builder, mlir::Region &region) {
  if (region.empty()) {
    region.emplaceBlock();
  }

  builder.setInsertionPointToEnd(&region.front());
}
} // namespace

mlir::LogicalResult
LowererImpl::lowerClassModification(const ClassModification &modification,
                                    mlir::Region &region) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  setInsertionPointToEnd(builder, region);

  for (const Child<Argument> &argument : modification.getArguments()) {
    if (mlir::failed(lowerArgument(*argument))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult
LowererImpl::lowerModification(const Modification &modification,
                               mlir::Region &region) {
  if (const ClassModification *classModification =
          modification.getClassModification()) {
    if (mlir::failed(lowerClassModification(*classModification, region))) {
      return mlir::failure();
    }
  }

  if (!modification.isBreak() && !modification.getValue()) {
    return mlir::success();
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  setInsertionPointToEnd(builder, region);
  mlir::Location location = loc(modification);

  if (modification.isBreak()) {
    builder.create<BindOp>(location, mlir::Value(), true);
    return mlir::success();
  }

  auto value = lowerExpression(*modification.getValue());

  if (mlir::failed(value)) {
    return mlir::failure();
  }

  builder.create<BindOp>(location, *value, false);
  return mlir::success();
}

mlir::LogicalResult LowererImpl::lowerArgument(const Argument &argument) {
  mlir::Location location = loc(argument);

  if (auto modification = argument.dyn_cast<ElementModification>()) {
    auto modifyOp = createOp<ModifyOp>(builder, location, 1);
    modifyOp.setNameAttr(getSymbolRef(modification->getName()));
    modifyOp.setIsEach(modification->isEach());
    modifyOp.setIsFinal(modification->isFinal());

    if (!modification->getDescription().empty()) {
      modifyOp.setDescriptionAttr(
          builder.getStringAttr(modification->getDescription()));
    }

    if (const Modification *nested = modification->getModification()) {
      return lowerModification(*nested, modifyOp.getModificationRegion());
    }

    return mlir::success();
  }

  if (auto redeclaration = argument.dyn_cast<ElementRedeclaration>()) {
    ElementPrefixes prefixes;
    prefixes.isRedeclare = redeclaration->isRedeclare();
    prefixes.isFinal = redeclaration->isFinal();
    prefixes.isReplaceable = redeclaration->isReplaceable();

    if (auto classDefinition =
            redeclaration->getElement()->dyn_cast<ClassDefinition>()) {
      return lowerClassDefinition(*classDefinition, nullptr, scope, prefixes,
                                  redeclaration->isEach(),
                                  redeclaration->getConstrainingClause());
    }

    auto clause = redeclaration->getElement()->cast<ComponentClause>();
    const ComponentDeclaration &declaration = *clause->getDeclarations()[0];

    Element component;
    component.kind = Element::Kind::Component;
    component.clause = clause;
    component.declaration = &declaration;
    component.owner = scope;

    return lowerComponent(*clause, declaration, prefixes,
                          redeclaration->isEach(), component,
                          redeclaration->getConstrainingClause());
  }

  auto deselection = argument.cast<InheritanceModification>();

  if (!deselection->isConnect()) {
    auto deselectOp = createOp<DeselectOp>(builder, location);
    deselectOp.setComponentAttr(mlir::FlatSymbolRefAttr::get(
        builder.getContext(), deselection->getComponent()));

    return mlir::success();
  }

  auto deselectOp = createOp<DeselectConnectOp>(builder, location, 1);

  if (mlir::failed(lowerIntoRegion(deselectOp.getBodyRegion(),
                                   {deselection->getConnectLhs(),
                                    deselection->getConnectRhs()},
                                   location))) {
    deselectOp.erase();
    return mlir::failure();
  }

  return mlir::success();
}
} // namespace marco::codegen::lowering::modelica
