#include "marco/Codegen/Lowering/Modelica/LowererImpl.h"

using namespace ::marco;
using namespace ::marco::ast::modelica;
using namespace ::marco::codegen::lowering::modelica;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering::modelica {
int64_t LowererImpl::getStaticDimension(const Expression &subscript,
                                        ClassNode *scope) {
  if (auto literal = subscript.dyn_cast<IntegerLiteral>()) {
    return literal->getValue();
  }

  // Dimensions indexed by Boolean or enumeration types.
  if (auto reference = subscript.dyn_cast<ComponentReference>()) {
    if (auto element = classTree.lookup(scope, reference->getName());
        element && element->isClass()) {
      ClassNode *classNode = element->classNode;

      // Follow aliases of enumeration types.
      while (classNode && classNode->getDefinition() &&
             classNode->getDefinition()->getForm() ==
                 ClassDefinition::Form::Short) {
        classNode = classTree.getShortClassBase(classNode);
      }

      if (classNode && classNode->getPredefined() == PredefinedClass::Boolean) {
        return 2;
      }

      if (classNode && classNode->isEnumeration() &&
          !classNode->getEnumerationLiterals().empty()) {
        return classNode->getEnumerationLiterals().size();
      }
    }
  }

  return mlir::ShapedType::kDynamic;
}

mlir::FailureOr<mlir::Type> LowererImpl::getClassType(ClassNode *classNode,
                                                      mlir::Location location) {
  if (!classNode) {
    return emitError(location, "unknown class");
  }

  if (auto it = classTypes.find(classNode); it != classTypes.end()) {
    return it->second;
  }

  if (!classTypesInProgress.insert(classNode).second) {
    return emitError(location, "cyclic definition of class '" +
                                   classNode->getPathString() + "'");
  }

  auto computeType = [&]() -> mlir::FailureOr<mlir::Type> {
    mlir::MLIRContext *context = builder.getContext();

    switch (classNode->getPredefined()) {
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
      return EnumerationType::get(context, getAbsoluteSymbolRef(classNode));
    case PredefinedClass::ExternalObject:
      return ObjectType::get(context, getAbsoluteSymbolRef(classNode));
    case PredefinedClass::Connections:
    case PredefinedClass::Clock:
      return emitError(location, "'" + classNode->getPathString() +
                                     "' can't be used as a type");
    case PredefinedClass::None:
      break;
    }

    const ClassDefinition *definition = classNode->getDefinition();

    switch (definition->getForm()) {
    case ClassDefinition::Form::Enumeration:
      return EnumerationType::get(context, getAbsoluteSymbolRef(classNode));

    case ClassDefinition::Form::Derivative:
      return emitError(location, "functions can't be used as types");

    case ClassDefinition::Form::Short: {
      auto baseType =
          getClassType(classTree.getShortClassBase(classNode), location);

      if (mlir::failed(baseType)) {
        return mlir::failure();
      }

      // The dimensions of the short class definition precede the ones of
      // the base class.
      llvm::SmallVector<int64_t> shape;

      for (const Child<Expression> &subscript : definition->getSubscripts()) {
        shape.push_back(
            getStaticDimension(*subscript, classNode->getParent()));
      }

      llvm::append_range(shape, getShape(*baseType));
      return getValueType(shape, mlir::getElementTypeOrSelf(*baseType));
    }

    case ClassDefinition::Form::Long:
    case ClassDefinition::Form::ClassExtends:
      break;
    }

    if (definition->getRestriction() == ClassRestriction::Function ||
        definition->getRestriction() == ClassRestriction::OperatorFunction) {
      return emitError(location, "functions can't be used as types");
    }

    // Classes extending a simple type without declaring components, such as
    // 'type T extends Real; end T;', have the values of the base type.
    llvm::SmallVector<ClassNode *> baseClasses =
        classTree.getBaseClasses(classNode);

    if (classNode->getComponents().empty() && baseClasses.size() == 1) {
      auto baseType = getClassType(baseClasses[0], location);

      if (mlir::succeeded(baseType) &&
          !mlir::isa<ObjectType>(mlir::getElementTypeOrSelf(*baseType))) {
        return baseType;
      }
    }

    return ObjectType::get(context, getAbsoluteSymbolRef(classNode));
  };

  auto type = computeType();
  classTypesInProgress.erase(classNode);

  if (mlir::failed(type)) {
    return mlir::failure();
  }

  classTypes[classNode] = *type;

  if (auto objectType = mlir::dyn_cast<ObjectType>(*type)) {
    objectClasses.try_emplace(objectType, classNode);
  }

  return *type;
}

mlir::FailureOr<mlir::Type>
LowererImpl::getComponentType(const Element &component) {
  std::pair<const ComponentDeclaration *, ClassNode *> key(
      component.declaration, component.accessScope);

  if (auto it = componentTypes.find(key); it != componentTypes.end()) {
    return it->second;
  }

  mlir::Location location = loc(*component.declaration);
  ClassNode *classNode = classTree.getComponentClass(component);

  if (!classNode) {
    return emitError(location,
                     "unknown class '" +
                         component.clause->getTypeSpecifier().toString() + "'");
  }

  auto classType = getClassType(classNode, location);

  if (mlir::failed(classType)) {
    return mlir::failure();
  }

  // The dimensions of the declaration precede the ones of the type
  // specifier, which precede the ones of the class.
  llvm::SmallVector<int64_t> shape;

  for (const Child<Expression> &subscript :
       component.declaration->getSubscripts()) {
    shape.push_back(getStaticDimension(*subscript, component.owner));
  }

  for (const Child<Expression> &subscript : component.clause->getSubscripts()) {
    shape.push_back(getStaticDimension(*subscript, component.owner));
  }

  llvm::append_range(shape, getShape(*classType));

  mlir::Type type =
      getValueType(shape, mlir::getElementTypeOrSelf(*classType));

  componentTypes[key] = type;
  return type;
}

ClassNode *LowererImpl::getObjectClass(mlir::Type type) {
  auto objectType =
      mlir::dyn_cast<ObjectType>(mlir::getElementTypeOrSelf(type));

  if (!objectType) {
    return nullptr;
  }

  if (auto it = objectClasses.find(objectType); it != objectClasses.end()) {
    return it->second;
  }

  llvm::SmallVector<llvm::StringRef> path;
  mlir::SymbolRefAttr name = objectType.getClassName();
  path.push_back(name.getRootReference().getValue());

  for (mlir::FlatSymbolRefAttr nested : name.getNestedReferences()) {
    path.push_back(nested.getValue());
  }

  return classTree.getClassByPath(path);
}

void LowererImpl::collectComponents(ClassNode *classNode,
                                    llvm::SmallVectorImpl<Element> &components) {
  if (!classNode || !classNode->getDefinition()) {
    return;
  }

  const ClassDefinition *definition = classNode->getDefinition();

  if (definition->getForm() == ClassDefinition::Form::Short) {
    collectComponents(classTree.getShortClassBase(classNode), components);
    return;
  }

  // The class inherited by a class-extends redeclaration comes first.
  if (definition->getForm() == ClassDefinition::Form::ClassExtends) {
    collectComponents(classTree.getInheritedClass(classNode), components);
  }

  // Inherited elements are placed at the position of the extends-clause,
  // preserving the declaration order.
  for (const Child<Node> &element : definition->getElements()) {
    if (auto extendsClause = element->dyn_cast<ExtendsClause>()) {
      collectComponents(classTree.resolveExtends(classNode, *extendsClause),
                        components);
      continue;
    }

    if (auto clause = element->dyn_cast<ComponentClause>()) {
      for (const Child<ComponentDeclaration> &declaration :
           clause->getDeclarations()) {
        components.push_back(
            classNode->getComponents().lookup(declaration->getName()));
      }
    }
  }
}

mlir::FailureOr<llvm::SmallVector<mlir::Type>>
LowererImpl::getFunctionResultTypes(ClassNode *function,
                                    mlir::Location location) {
  llvm::SmallVector<Element> components;
  collectComponents(function, components);

  llvm::SmallVector<mlir::Type> types;

  for (const Element &component : components) {
    if (component.clause->getTypePrefix().causality != ast::modelica::Causality::Output) {
      continue;
    }

    auto type = getComponentType(component);

    if (mlir::failed(type)) {
      return mlir::failure();
    }

    types.push_back(*type);
  }

  return types;
}
} // namespace marco::codegen::lowering::modelica
