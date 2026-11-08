#include "marco/Dialect/Modelica/Transforms/Instantiation/Instantiator.h"
#include "marco/Dialect/Modelica/IR/ConstantFolding.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ScopeExit.h"

using namespace ::mlir::modelica;
using namespace ::mlir::modelica::instantiation;

namespace {
/// Get the name of an element declaration.
llvm::StringRef getSymbolName(mlir::Operation *op) {
  return op
      ->getAttrOfType<mlir::StringAttr>(mlir::SymbolTable::getSymbolAttrName())
      .getValue();
}

bool hasFlag(mlir::Operation *op, llvm::StringRef name) {
  return static_cast<bool>(op->getAttr(name));
}

/// Get the textual representation of a class-level expression, used to
/// compare connections syntactically.
std::string getExpressionText(mlir::Value value) {
  mlir::Operation *op = value.getDefiningOp();

  if (!op) {
    return "?";
  }

  std::string text = op->getName().getStringRef().str();
  llvm::raw_string_ostream os(text);
  os << op->getRawDictionaryAttrs();

  if (auto properties = op->getPropertiesAsAttribute()) {
    os << properties;
  }

  os << "(";

  for (mlir::Value operand : op->getOperands()) {
    os << getExpressionText(operand) << ",";
  }

  os << ")";
  return text;
}
} // namespace

namespace mlir::modelica::instantiation {
std::string getReferenceText(mlir::Value value) {
  return getExpressionText(value);
}
} // namespace mlir::modelica::instantiation

namespace {
/// Get the most restrictive of two variabilities.
std::optional<Variability> combine(std::optional<Variability> lhs,
                                   std::optional<Variability> rhs) {
  if (!lhs) {
    return rhs;
  }

  if (!rhs) {
    return lhs;
  }

  return static_cast<uint32_t>(*lhs) > static_cast<uint32_t>(*rhs) ? lhs : rhs;
}
} // namespace

namespace mlir::modelica::instantiation {
Instantiator::Instantiator(mlir::ModuleOp module)
    : module(module), builder(module.getContext()) {
  mlir::MLIRContext *context = module.getContext();

  rootScope = createScope(module, nullptr, {}, nullptr, nullptr);

  for (mlir::Operation &op : module.getBody()->getOperations()) {
    if (mlir::isa<ClassOp, EnumerationOp>(op)) {
      classOps.push_back(&op);
    }
  }

  scratchModule = mlir::ModuleOp::create(module.getLoc());
  mlir::OpBuilder scratchBuilder(context);
  scratchBuilder.setInsertionPointToStart(scratchModule->getBody());

  unknownIndex = scratchBuilder
                     .create<mlir::UnrealizedConversionCastOp>(
                         module.getLoc(), IntegerType::get(context),
                         mlir::ValueRange())
                     .getResult(0);
}

Instantiator::~Instantiator() = default;

//===---------------------------------------------------------------------===//
// Scopes
//===---------------------------------------------------------------------===//

Scope *Instantiator::createScope(mlir::Operation *op, Scope *parent,
                                 Modifier modifier, Component *instance,
                                 Element *replaced) {
  auto &scope = scopes.emplace_back(std::make_unique<Scope>());
  scope->op = op;
  scope->parent = parent;
  scope->modifier = std::move(modifier);
  scope->instance = instance;
  scope->replaced = replaced;

  if (op && !mlir::isa<mlir::ModuleOp>(op)) {
    llvm::StringRef name = getSymbolName(op);

    if (parent && !parent->path.empty()) {
      scope->path = parent->path + "." + name.str();
    } else {
      scope->path = name.str();
    }
  }

  return scope.get();
}

Scope *Instantiator::createPredefinedScope(PredefinedClass predefined,
                                           Modifier modifier,
                                           Component *instance) {
  Scope *scope = createScope(nullptr, rootScope, std::move(modifier), instance,
                             nullptr);

  scope->predefined = predefined;
  scope->state = Scope::State::Expanded;

  switch (predefined) {
  case PredefinedClass::Real:
    scope->path = "Real";
    break;
  case PredefinedClass::Integer:
    scope->path = "Integer";
    break;
  case PredefinedClass::Boolean:
    scope->path = "Boolean";
    break;
  case PredefinedClass::String:
    scope->path = "String";
    break;
  case PredefinedClass::StateSelect:
    scope->path = "StateSelect";
    scope->literals = {"never", "avoid", "default", "prefer", "always"};
    break;
  case PredefinedClass::AssertionLevel:
    scope->path = "AssertionLevel";
    scope->literals = {"warning", "error"};
    break;
  case PredefinedClass::ExternalObject:
    scope->path = "ExternalObject";
    break;
  case PredefinedClass::Connections:
    scope->path = "Connections";
    break;
  case PredefinedClass::Clock:
    scope->path = "Clock";
    break;
  case PredefinedClass::None:
    break;
  }

  switch (predefined) {
  case PredefinedClass::Real:
  case PredefinedClass::Integer:
  case PredefinedClass::Boolean:
  case PredefinedClass::String:
  case PredefinedClass::StateSelect:
  case PredefinedClass::AssertionLevel:
    scope->primitiveBase = scope;
    break;
  default:
    break;
  }

  return scope;
}

mlir::LogicalResult Instantiator::createLocals(Scope *scope) {
  if (scope->state != Scope::State::Created) {
    return mlir::success(scope->state != Scope::State::Failed);
  }

  scope->state = Scope::State::LocalsCreated;

  auto addElement = [&](Element::Kind kind, mlir::Operation *op,
                        PredefinedClass predefined = PredefinedClass::None) {
    auto &element = elements.emplace_back(std::make_unique<Element>());
    element->kind = kind;
    element->op = op;
    element->predefined = predefined;
    element->declScope = scope;
    element->owner = scope;

    llvm::StringRef name = element->getName();

    if (const Modifier *modifier = scope->modifier.getChild(name)) {
      element->modifier = *modifier;
    }

    // Redeclarations by the modification of the class.
    if (auto it = scope->modifier.redeclarations.find(name.str());
        it != scope->modifier.redeclarations.end()) {
      element->op = it->second.op;
      element->declScope = it->second.scope;

      if (auto classOp = mlir::dyn_cast<ClassOp>(element->op);
          classOp && classOp.getIsClassExtends()) {
        auto &original = elements.emplace_back(std::make_unique<Element>());
        original->kind = kind;
        original->op = op;
        original->declScope = scope;
        original->owner = scope;
        element->replaced = original.get();
      }
    }

    scope->localElements.try_emplace(name, element.get());
  };

  if (scope->isRoot()) {
    for (mlir::Operation &op : module.getBody()->getOperations()) {
      if (mlir::isa<ClassOp, EnumerationOp>(op)) {
        addElement(Element::Kind::Class, &op);
      }
    }

    // Predefined classes.
    for (PredefinedClass predefined :
         {PredefinedClass::Real, PredefinedClass::Integer,
          PredefinedClass::Boolean, PredefinedClass::String,
          PredefinedClass::StateSelect, PredefinedClass::AssertionLevel,
          PredefinedClass::ExternalObject, PredefinedClass::Connections,
          PredefinedClass::Clock}) {
      addElement(Element::Kind::Class, nullptr, predefined);
    }

    scope->state = Scope::State::Expanded;
    return mlir::success();
  }

  if (auto enumerationOp = mlir::dyn_cast<EnumerationOp>(scope->op)) {
    for (auto literalOp : enumerationOp.getBody()->getOps<EnumLiteralOp>()) {
      scope->literals.push_back(literalOp.getSymName().str());
    }

    scope->primitiveBase = scope;
    scope->state = Scope::State::Expanded;
    return mlir::success();
  }

  auto classOp = mlir::cast<ClassOp>(scope->op);

  for (mlir::Operation &op : classOp.getBody()->getOperations()) {
    if (auto importOp = mlir::dyn_cast<ImportOp>(op)) {
      scope->imports.push_back(importOp);
      continue;
    }

    if (!mlir::isa<ClassOp, EnumerationOp, ComponentOp>(op)) {
      continue;
    }

    // Redeclarations of inherited elements are applied to the base classes.
    if (hasFlag(&op, "isRedeclare") || hasFlag(&op, "isClassExtends")) {
      continue;
    }

    addElement(mlir::isa<ComponentOp>(op) ? Element::Kind::Component
                                          : Element::Kind::Class,
               &op);
  }

  return mlir::success();
}

mlir::LogicalResult Instantiator::expand(Scope *scope) {
  switch (scope->state) {
  case Scope::State::Expanded:
    return mlir::success();
  case Scope::State::Failed:
    return mlir::failure();
  case Scope::State::Expanding:
    return mlir::emitError(scope->getLoc())
           << "class '" << scope->path << "' extends itself";
  default:
    break;
  }

  if (mlir::failed(createLocals(scope))) {
    return mlir::failure();
  }

  if (scope->state == Scope::State::Expanded) {
    return mlir::success();
  }

  scope->state = Scope::State::Expanding;
  auto classOp = mlir::cast<ClassOp>(scope->op);

  // The redeclarations within the body of the class apply to the inherited
  // elements, but are overridden by the modification of the class.
  Modifier bodyRedeclarations;

  for (mlir::Operation &op : classOp.getBody()->getOperations()) {
    if (mlir::isa<ClassOp, EnumerationOp, ComponentOp>(op) &&
        (hasFlag(&op, "isRedeclare") || hasFlag(&op, "isClassExtends"))) {
      bodyRedeclarations.redeclarations[getSymbolName(&op).str()] =
          Redeclaration{&op, scope};
    }
  }

  Modifier outerModifier = Modifier::merge(scope->modifier, bodyRedeclarations);

  auto addElement = [&](Element *element) {
    llvm::StringRef name = element->getName();

    if (auto it = scope->elementsByName.find(name);
        it != scope->elementsByName.end()) {
      // Local elements take precedence over the inherited ones, while
      // elements inherited multiple times are kept once.
      if (scope->localElements.lookup(name) == element) {
        auto pos = llvm::find(scope->elements, it->second);
        *pos = element;
        it->second = element;
      }

      return;
    }

    scope->elements.push_back(element);
    scope->elementsByName[name] = element;
  };

  auto fail = [&]() {
    scope->state = Scope::State::Failed;
    return mlir::failure();
  };

  for (mlir::Operation &op : classOp.getBody()->getOperations()) {
    if (auto extendsOp = mlir::dyn_cast<ExtendsOp>(op)) {
      Element *baseElement = nullptr;

      if (auto baseClass = extendsOp.getBaseClassAttr()) {
        baseElement = lookupName(scope, baseClass, /*skipInherited=*/true);

        if (!baseElement) {
          mlir::emitError(extendsOp.getLoc())
              << "unknown base class '" << baseClass << "'";
          return fail();
        }
      } else {
        baseElement = scope->replaced;

        if (!baseElement) {
          mlir::emitError(extendsOp.getLoc())
              << "class-extends redeclaration of '" << scope->path
              << "' without inherited class";
          return fail();
        }
      }

      if (!baseElement->isClass()) {
        mlir::emitError(extendsOp.getLoc())
            << "base class '" << baseElement->getName() << "' is a component";
        return fail();
      }

      Modifier extendsModifier =
          Modifier::parse(extendsOp.getModificationRegion(), scope);

      Scope *base = instantiateClass(
          baseElement, Modifier::merge(outerModifier, extendsModifier),
          scope->instance);

      if (mlir::failed(expand(base))) {
        return fail();
      }

      base->derived = scope;
      llvm::StringSet<> deselected;

      if (!extendsOp.getModificationRegion().empty()) {
        mlir::Block &modification = extendsOp.getModificationRegion().front();

        for (auto deselectOp : modification.getOps<DeselectOp>()) {
          deselected.insert(deselectOp.getComponent());
        }

        for (auto deselectConnectOp :
             modification.getOps<DeselectConnectOp>()) {
          auto values = mlir::cast<YieldOp>(
                            deselectConnectOp.getBodyRegion().front().getTerminator())
                            .getValues();

          base->deselectedConnections.emplace(getReferenceText(values[0]),
                                              getReferenceText(values[1]));
        }
      }

      for (Element *element : base->elements) {
        if (!deselected.contains(element->getName())) {
          addElement(element);
        }
      }

      scope->bases.push_back(base);

      for (const auto &name : deselected) {
        scope->deselected.insert(name.getKey());
      }

      for (const auto &name : base->deselected) {
        scope->deselected.insert(name.getKey());
      }

      scope->sections.append(base->sections.begin(), base->sections.end());

      if (base->isPrimitive()) {
        scope->primitiveBase = base->primitiveBase;
        scope->literals = base->literals;

        if (!extendsOp.getDimensions().empty() || extendsOp.getCausality()) {
          scope->shortClassExtends.emplace_back(extendsOp, scope);
        }

        scope->shortClassExtends.append(base->shortClassExtends.begin(),
                                        base->shortClassExtends.end());
      }

      continue;
    }

    if (mlir::isa<ClassOp, EnumerationOp, ComponentOp>(op)) {
      if (!hasFlag(&op, "isRedeclare") && !hasFlag(&op, "isClassExtends")) {
        if (Element *element =
                scope->localElements.lookup(getSymbolName(&op))) {
          addElement(element);
        }
      }

      continue;
    }

    if (mlir::isa<EquationSectionOp, AlgorithmSectionOp, ExternalOp>(op)) {
      scope->sections.emplace_back(&op, scope);
    }
  }

  scope->state = Scope::State::Expanded;
  return mlir::success();
}

mlir::SymbolRefAttr Instantiator::getClassSymbol(Scope *scope) {
  mlir::MLIRContext *context = module.getContext();

  if (!scope->op) {
    return mlir::FlatSymbolRefAttr::get(context, scope->path);
  }

  llvm::SmallVector<llvm::StringRef> names;

  for (mlir::Operation *op = scope->op; op && !mlir::isa<mlir::ModuleOp>(op);
       op = op->getParentOp()) {
    if (mlir::isa<ClassOp, EnumerationOp>(op)) {
      names.push_back(getSymbolName(op));
    }
  }

  std::reverse(names.begin(), names.end());
  llvm::SmallVector<mlir::FlatSymbolRefAttr> nested;

  for (llvm::StringRef name : llvm::drop_begin(names)) {
    nested.push_back(mlir::FlatSymbolRefAttr::get(context, name));
  }

  return mlir::SymbolRefAttr::get(context, names.front(), nested);
}

//===---------------------------------------------------------------------===//
// Lookup
//===---------------------------------------------------------------------===//

Element *Instantiator::lookupSimple(Scope *scope, llvm::StringRef name,
                                    bool skipInherited) {
  auto lookupPredefined = [&]() -> Element * {
    Element *element = rootScope->localElements.lookup(name);
    return element && !element->op ? element : nullptr;
  };

  for (Scope *current = scope; current; current = current->parent) {
    if (current->isRoot()) {
      (void)createLocals(current);
      return current->localElements.lookup(name);
    }

    if (mlir::failed(createLocals(current))) {
      return nullptr;
    }

    if ((current == scope && skipInherited) ||
        current->state == Scope::State::Expanding) {
      // Only the elements known so far are considered.
      if (Element *element = current->localElements.lookup(name)) {
        return element;
      }

      if (Element *element = current->elementsByName.lookup(name)) {
        return element;
      }
    } else {
      if (mlir::failed(expand(current))) {
        return nullptr;
      }

      if (Element *element = current->elementsByName.lookup(name)) {
        return element;
      }
    }

    // Import-clauses.
    for (ImportOp importOp : current->imports) {
      mlir::SymbolRefAttr path = importOp.getPath();

      if (auto alias = importOp.getAliasAttr()) {
        if (alias.getValue() == name) {
          return lookupName(rootScope, path);
        }

        continue;
      }

      if (importOp.getIsUnqualified()) {
        Element *package = lookupName(rootScope, path);

        if (package && package->isClass()) {
          if (Element *element = lookupMember(getClassScope(package), name)) {
            return element;
          }
        }

        continue;
      }

      if (auto names = importOp.getNamesAttr()) {
        bool imported = llvm::any_of(names, [&](mlir::Attribute attr) {
          return mlir::cast<mlir::FlatSymbolRefAttr>(attr).getValue() == name;
        });

        if (imported) {
          Element *package = lookupName(rootScope, path);

          if (package && package->isClass()) {
            return lookupMember(getClassScope(package), name);
          }
        }

        continue;
      }

      if (path.getLeafReference() == name) {
        return lookupName(rootScope, path);
      }
    }

    // Encapsulated classes only see the predefined classes.
    if (current->isEncapsulated()) {
      return lookupPredefined();
    }
  }

  return nullptr;
}

Element *Instantiator::lookupName(Scope *scope, mlir::SymbolRefAttr name,
                                  bool skipInherited) {
  llvm::SmallVector<llvm::StringRef> parts;
  parts.push_back(name.getRootReference().getValue());

  for (mlir::FlatSymbolRefAttr nested : name.getNestedReferences()) {
    parts.push_back(nested.getValue());
  }

  Element *element = nullptr;
  size_t index = 0;

  if (parts.front() == ".") {
    // Global name.
    if (parts.size() < 2) {
      return nullptr;
    }

    (void)createLocals(rootScope);
    element = rootScope->localElements.lookup(parts[1]);
    index = 2;
  } else {
    element = lookupSimple(scope, parts.front(), skipInherited);
    index = 1;
  }

  for (; index < parts.size() && element; ++index) {
    Scope *memberScope = nullptr;

    if (element->isClass()) {
      memberScope = getClassScope(element);
    } else if (Component *component = getComponent(element)) {
      memberScope = component->classScope;
    }

    if (!memberScope) {
      return nullptr;
    }

    element = lookupMember(memberScope, parts[index]);
  }

  return element;
}

Element *Instantiator::lookupMember(Scope *scope, llvm::StringRef name) {
  // Classes being expanded, such as packages extending one of their nested
  // classes, only provide the elements known so far.
  if (scope->state == Scope::State::Expanding) {
    if (Element *element = scope->localElements.lookup(name)) {
      return element;
    }

    return scope->elementsByName.lookup(name);
  }

  if (mlir::failed(expand(scope))) {
    return nullptr;
  }

  if (scope->isRoot()) {
    return scope->localElements.lookup(name);
  }

  return scope->elementsByName.lookup(name);
}

Scope *Instantiator::getClassScope(Element *element) {
  if (!element->classScope) {
    element->classScope = instantiateClass(element, {}, nullptr);
  }

  return element->classScope;
}

Scope *Instantiator::instantiateClass(Element *element, Modifier modifier,
                                      Component *instance) {
  if (element->predefined != PredefinedClass::None) {
    return createPredefinedScope(element->predefined, std::move(modifier),
                                 instance);
  }

  return createScope(element->op, element->declScope, std::move(modifier),
                     instance, element->replaced);
}

//===---------------------------------------------------------------------===//
// Components
//===---------------------------------------------------------------------===//

Component *Instantiator::getComponent(Element *element) {
  if (element->component) {
    return element->component->classScope ? element->component : nullptr;
  }

  auto componentOp = mlir::cast<ComponentOp>(element->op);
  auto &component = components.emplace_back(std::make_unique<Component>());
  Component *parent = element->owner->instance;

  component->element = element;
  component->parent = parent;
  component->name = element->getName().str();
  component->numDeclaredDims = componentOp.getDimensions().size();
  component->indexOffset = parent ? parent->getNumIndices() : 0;

  component->isProtected =
      componentOp.getIsProtected() ||
      (parent && !parent->isRoot() && parent->isProtected);

  component->variability = componentOp.getVariability();
  component->causality = componentOp.getCausality();

  if (parent && !parent->isRoot()) {
    // The prefixes of structured components apply to their elements.
    component->variability =
        combine(component->variability, parent->variability);

    if (!component->causality) {
      component->causality = parent->causality;
    }
  }

  element->component = component.get();

  if (mlir::failed(instantiateComponent(component.get()))) {
    return nullptr;
  }

  return component.get();
}

mlir::LogicalResult Instantiator::instantiateComponent(Component *component) {
  Element *element = component->element;
  ComponentOp componentOp = component->getDeclaration();

  if (!activeInstantiations.insert(component).second) {
    return mlir::emitError(componentOp.getLoc())
           << "recursive instantiation of component '" << component->name
           << "'";
  }

  auto guard = llvm::make_scope_exit(
      [&]() { activeInstantiations.erase(component); });

  Element *classElement =
      lookupName(element->declScope, componentOp.getClassNameAttr());

  if (!classElement || !classElement->isClass()) {
    return mlir::emitError(componentOp.getLoc())
           << "unknown class '" << componentOp.getClassNameAttr()
           << "' of component '" << component->name << "'";
  }

  Scope *unmodifiedClass = getClassScope(classElement);

  if (mlir::failed(expand(unmodifiedClass))) {
    return mlir::failure();
  }

  component->modifier = Modifier::merge(
      element->modifier,
      Modifier::parse(componentOp.getModificationRegion(), element->declScope));

  // The modifications of arrays of instances apply to their elements.
  if (!unmodifiedClass->isPrimitive() && component->numDeclaredDims != 0) {
    llvm::SmallVector<unsigned> positions;

    for (unsigned i = 0; i < component->numDeclaredDims; ++i) {
      positions.push_back(component->indexOffset + i);
    }

    if (!component->modifier.isEach) {
      if (component->modifier.value) {
        component->modifier.value->splitPositions.append(positions);
      }

      for (auto &child : llvm::make_second_range(component->modifier.children)) {
        child.split(positions);
      }
    }
  }

  Scope *classScope =
      instantiateClass(classElement, component->modifier, component);

  if (mlir::failed(expand(classScope))) {
    return mlir::failure();
  }

  if (!component->causality) {
    for (auto [extendsOp, scope] : classScope->shortClassExtends) {
      if (auto causality = extendsOp.getCausality()) {
        component->causality = causality;
        break;
      }
    }
  }

  component->classScope = classScope;
  return mlir::success();
}

mlir::FailureOr<int64_t> Instantiator::evaluateDimension(
    DimensionAttr dimension, mlir::Value value, Scope *scope,
    Component *component, mlir::Location loc) {
  switch (dimension.getKind()) {
  case DimensionKind::fixed:
    return dimension.getSize();

  case DimensionKind::boolean:
    return 2;

  case DimensionKind::enumeration: {
    Element *element = lookupName(scope, dimension.getEnumeration());

    if (!element || !element->isClass()) {
      return mlir::emitError(loc)
             << "unknown enumeration '" << dimension.getEnumeration() << "'";
    }

    Scope *enumeration = getClassScope(element);

    if (mlir::failed(expand(enumeration)) || !enumeration->isPrimitive()) {
      return mlir::failure();
    }

    return static_cast<int64_t>(enumeration->literals.size());
  }

  case DimensionKind::expression: {
    auto result = evaluateStructural(value, scope, component, loc);

    if (mlir::failed(result)) {
      return mlir::failure();
    }

    std::optional<int64_t> size = getIntegerValue(*result);

    if (!size) {
      return mlir::failure();
    }

    return *size;
  }

  case DimensionKind::colon:
    break;
  }

  return mlir::failure();
}

mlir::FailureOr<llvm::ArrayRef<int64_t>>
Instantiator::getDims(Component *component) {
  if (component->dims) {
    return llvm::ArrayRef<int64_t>(*component->dims);
  }

  ComponentOp componentOp = component->getDeclaration();

  Component *root = component;

  while (root->parent) {
    root = root->parent;
  }

  bool allowDynamic = root != modelRoot;

  if (!activeDimensions.insert(component).second) {
    // Within functions, the dimensions of the inputs may refer to the
    // inputs themselves (e.g., 'Real A[:, size(A, 1)]'), and are thus
    // dynamic.
    if (allowDynamic) {
      return mlir::failure();
    }

    return mlir::emitError(componentOp.getLoc())
           << "the dimensions of '" << component->getPath()
           << "' depend on themselves";
  }

  auto guard =
      llvm::make_scope_exit([&]() { activeDimensions.erase(component); });
  llvm::SmallVector<int64_t> dims;

  // Sizes of the dimensions given by the binding, used for the colon
  // dimensions.
  std::optional<llvm::SmallVector<int64_t>> bindingShape;

  auto getBindingShape = [&]() -> std::optional<llvm::SmallVector<int64_t>> {
    if (bindingShape) {
      return bindingShape;
    }

    const Modifier &modifier = component->isPrimitive()
                                   ? component->classScope->primitiveBase->modifier
                                   : component->modifier;

    const ModifierValue *value = modifier.getValue();

    if (!value) {
      return std::nullopt;
    }

    auto scratch = createScratchArea();
    IndexValues indices(component->indexOffset, nullptr);
    auto emitted = emitModifierValue(*value, indices, scratch->builder);

    if (mlir::failed(emitted)) {
      return std::nullopt;
    }

    llvm::SmallVector<int64_t> shape(modelica::getShape(emitted->getType()));

    if (mlir::ShapedType::isDynamicShape(shape)) {
      if (auto attr = mlir::dyn_cast_or_null<mlir::TypedAttr>(
              evaluateEmitted(*emitted))) {
        shape.assign(modelica::getShape(attr.getType()).begin(),
                     modelica::getShape(attr.getType()).end());
      }
    }

    bindingShape = shape;
    return bindingShape;
  };

  auto addDimensions = [&](mlir::ArrayAttr dimensions, mlir::Region &region,
                           Scope *scope,
                           mlir::Location loc) -> mlir::LogicalResult {
    mlir::ValueRange values;

    if (!region.empty()) {
      values = mlir::cast<YieldOp>(region.front().getTerminator()).getValues();
    }

    size_t valuePos = 0;

    for (mlir::Attribute attr : dimensions) {
      auto dimension = mlir::cast<DimensionAttr>(attr);
      mlir::Value value;

      if (dimension.getKind() == DimensionKind::expression) {
        value = values[valuePos++];
      }

      if (dimension.getKind() == DimensionKind::colon) {
        auto shape = getBindingShape();
        size_t pos = dims.size();

        if (shape && pos < shape->size() &&
            !mlir::ShapedType::isDynamic((*shape)[pos])) {
          dims.push_back((*shape)[pos]);
          continue;
        }

        if (allowDynamic) {
          dims.push_back(mlir::ShapedType::kDynamic);
          continue;
        }

        return mlir::emitError(loc)
               << "can't determine the size of the dimension " << pos + 1
               << " of '" << component->getPath() << "'";
      }

      auto size = evaluateDimension(dimension, value, scope, component, loc);

      if (mlir::succeeded(size)) {
        dims.push_back(*size);
        continue;
      }

      if (allowDynamic) {
        dims.push_back(mlir::ShapedType::kDynamic);
        continue;
      }

      return mlir::emitError(loc)
             << "can't evaluate the size of the dimension " << dims.size() + 1
             << " of '" << component->getPath() << "'";
    }

    return mlir::success();
  };

  if (mlir::failed(addDimensions(componentOp.getDimensions(),
                                 componentOp.getDimensionsRegion(),
                                 component->element->declScope,
                                 componentOp.getLoc()))) {
    return mlir::failure();
  }

  if (component->isPrimitive()) {
    for (auto [extendsOp, scope] : component->classScope->shortClassExtends) {
      if (mlir::failed(addDimensions(
              extendsOp.getDimensions(), extendsOp.getDimensionsRegion(),
              scope, extendsOp.getLoc()))) {
        return mlir::failure();
      }
    }
  }

  component->dims = std::move(dims);
  return llvm::ArrayRef<int64_t>(*component->dims);
}

mlir::FailureOr<bool> Instantiator::isEnabled(Component *component) {
  if (component->enabled) {
    return *component->enabled;
  }

  ComponentOp componentOp = component->getDeclaration();

  if (!componentOp || componentOp.getConditionRegion().empty()) {
    component->enabled = true;
    return true;
  }

  mlir::Value condition =
      mlir::cast<YieldOp>(
          componentOp.getConditionRegion().front().getTerminator())
          .getValues()[0];

  auto result = evaluateStructural(condition, component->element->declScope,
                                   component, componentOp.getLoc());

  if (mlir::failed(result)) {
    return mlir::failure();
  }

  std::optional<bool> value = getBooleanValue(*result);

  if (!value) {
    return mlir::emitError(componentOp.getLoc())
           << "can't evaluate the condition of the conditional component '"
           << component->getPath() << "'";
  }

  component->enabled = *value;
  return *value;
}

Component *Instantiator::resolveOuter(Component *component) {
  if (component->inner) {
    return component->inner;
  }

  llvm::StringRef name = component->name;
  Component *root = component;

  while (root->parent) {
    root = root->parent;
  }

  // The inner element is searched in the instances enclosing the one
  // containing the outer element.
  for (Component *instance =
           component->parent ? component->parent->parent : nullptr;
       instance; instance = instance->parent) {
    if (Element *element = lookupMember(instance->classScope, name);
        element && element->isComponent()) {
      Component *candidate = getComponent(element);

      if (candidate && candidate->isInner()) {
        component->inner = candidate;
        return candidate;
      }
    }
  }

  for (Element *element : root->implicitInners) {
    if (element->getName() == name) {
      component->inner = getComponent(element);
      return component->inner;
    }
  }

  // Missing inner element: an implicit one is added to the root.
  ComponentOp componentOp = component->getDeclaration();

  mlir::emitWarning(componentOp.getLoc())
      << "no inner declaration for the outer component '"
      << component->getPath() << "', adding one at the top level";

  auto &element = elements.emplace_back(std::make_unique<Element>());
  element->kind = Element::Kind::Component;
  element->op = componentOp;
  element->declScope = component->element->declScope;
  element->owner = root->classScope;
  element->isImplicitInner = true;
  root->implicitInners.push_back(element.get());

  component->inner = getComponent(element.get());
  return component->inner;
}

mlir::Type Instantiator::getScalarType(Scope *primitive) {
  mlir::MLIRContext *context = module.getContext();

  switch (primitive->predefined) {
  case PredefinedClass::Real:
    return RealType::get(context);
  case PredefinedClass::Integer:
    return IntegerType::get(context);
  case PredefinedClass::Boolean:
    return BooleanType::get(context);
  case PredefinedClass::String:
    return StringType::get(context);
  case PredefinedClass::StateSelect:
  case PredefinedClass::AssertionLevel:
    return EnumerationType::get(context, getClassSymbol(primitive));
  default:
    break;
  }

  if (mlir::isa_and_nonnull<EnumerationOp>(primitive->op)) {
    return EnumerationType::get(context, getClassSymbol(primitive));
  }

  return nullptr;
}

mlir::FailureOr<mlir::Type> Instantiator::getType(Component *component) {
  auto dims = getDims(component);

  if (mlir::failed(dims)) {
    return mlir::failure();
  }

  mlir::Type elementType;

  if (component->isPrimitive()) {
    elementType = getScalarType(component->classScope->primitiveBase);
  } else {
    elementType = ObjectType::get(module.getContext(),
                                  getClassSymbol(component->classScope));
  }

  return getValueType(*dims, elementType);
}
} // namespace mlir::modelica::instantiation
